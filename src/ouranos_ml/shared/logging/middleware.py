"""ASGI middleware that logs one structured event per HTTP request completion."""

import re
import time
from uuid import uuid4

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from structlog.contextvars import bind_contextvars, clear_contextvars

logger = structlog.get_logger(__name__)

_REQUEST_ID_HEADER = b"x-request-id"
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")


def _request_id(scope: Scope) -> str:
    """Reuse a well-formed incoming ``X-Request-ID``, otherwise generate one."""
    for name, value in scope.get("headers", []):
        if name.lower() == _REQUEST_ID_HEADER:
            candidate = value.decode("latin-1")
            if _VALID_REQUEST_ID.fullmatch(candidate):
                return candidate
            break
    return uuid4().hex


def _record_exception_on_span(exc: Exception) -> None:
    """Mark the active span failed, as the OTel FastAPI instrumentation would if the exception reached it."""
    span = trace.get_current_span()
    if span.is_recording():
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))


class RequestLoggingMiddleware:
    """Pure ASGI middleware emitting a completion event for every HTTP request.

    Uses a raw ASGI wrapper (not BaseHTTPMiddleware) so streaming responses such
    as the chat completions SSE endpoint are unaffected. Every HTTP response,
    including excluded paths, carries the request id as ``x-request-id``.

    An unhandled exception raised before the response starts is answered here
    with an OpenAI-style JSON 500 instead of being re-raised, so the error body
    still gets the request id header and the outer CORS headers.
    """

    EXCLUDED_PATHS: frozenset[str] = frozenset({"/health"})

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the next ASGI application in the stack."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Tag the request with an id, log its outcome, and turn unhandled exceptions into a JSON 500."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id(scope)
        scope.setdefault("state", {})["request_id"] = request_id
        clear_contextvars()
        bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        status_code: list[int] = []

        async def send_wrapper(message: Message) -> None:
            if message.get("type") == "http.response.start":
                status_code.append(message["status"])
                message["headers"] = [*message.get("headers", []), (_REQUEST_ID_HEADER, request_id.encode("latin-1"))]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            logger.error("request failed", method=scope["method"], path=scope["path"], exc_info=True)
            if status_code:
                raise
            _record_exception_on_span(exc)
            response = JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Internal server error",
                        "type": "server_error",
                        "request_id": request_id,
                    },
                },
            )
            await response(scope, receive, send_wrapper)
        else:
            if scope["path"] not in self.EXCLUDED_PATHS:
                logger.info(
                    "request completed",
                    method=scope["method"],
                    path=scope["path"],
                    status_code=status_code[0] if status_code else None,
                    duration_ms=round((time.perf_counter() - start) * 1000, 2),
                )
        finally:
            clear_contextvars()
