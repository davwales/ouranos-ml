"""SwarmUI failures and their translation into HTTP responses."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class SwarmUIError(Exception):
    """A SwarmUI call failed; ``message`` is SwarmUI's own text, which is fit to show to end users."""

    status_code: int = 502
    error_type: str = "upstream_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SwarmUINotConfiguredError(SwarmUIError):
    """No SwarmUI base URL is configured."""

    status_code = 503
    error_type = "not_configured"


class SwarmUIUnavailableError(SwarmUIError):
    """SwarmUI could not be reached, or dropped the connection."""

    error_type = "server_error"


class SwarmUIModelNotFoundError(SwarmUIError):
    """The requested model does not exist on the SwarmUI server."""

    status_code = 404
    error_type = "model_not_found"


class SwarmUIInvalidSessionError(SwarmUIError):
    """SwarmUI no longer knows the session (it expired, or SwarmUI restarted)."""


class SwarmUITimeoutError(SwarmUIError):
    """SwarmUI did not answer, or did not finish a generation, in time."""

    status_code = 504
    error_type = "timeout"


async def _swarmui_error(_: Request, exc: Exception) -> JSONResponse:
    """Render a SwarmUI failure as an OpenAI-style error body with the failure's status code."""
    assert isinstance(exc, SwarmUIError)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_type,
            },
        },
    )


def register_swarmui_error_handler(app: FastAPI) -> None:
    """Handle ``SwarmUIError`` (and its subclasses) app-wide."""
    app.add_exception_handler(SwarmUIError, _swarmui_error)
