"""ASGI middleware that logs one structured event per HTTP request completion."""

import time
from collections.abc import MutableMapping
from uuid import uuid4

import structlog
from starlette.types import ASGIApp, Receive, Scope, Send
from structlog.contextvars import bind_contextvars, clear_contextvars

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware:
    """Pure ASGI middleware emitting a completion event for every HTTP request.

    Uses a raw ASGI wrapper (not BaseHTTPMiddleware) so streaming responses such
    as the chat completions SSE endpoint are unaffected.
    """

    EXCLUDED_PATHS: frozenset[str] = frozenset({"/health"})

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the next ASGI application in the stack."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Log a completion event unless the request type or path is excluded."""
        if scope["type"] != "http" or scope["path"] in self.EXCLUDED_PATHS:
            await self.app(scope, receive, send)
            return

        clear_contextvars()
        bind_contextvars(request_id=uuid4().hex)
        start = time.perf_counter()
        status_code: list[int] = []

        async def send_wrapper(message: MutableMapping[str, object]) -> None:
            if message.get("type") == "http.response.start":
                status = message.get("status")
                if isinstance(status, int):
                    status_code.append(status)
            await send(message)  # type: ignore[arg-type]

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.error("request failed", method=scope["method"], path=scope["path"], exc_info=True)
            raise
        else:
            logger.info(
                "request completed",
                method=scope["method"],
                path=scope["path"],
                status_code=status_code[0] if status_code else None,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )
        finally:
            clear_contextvars()
