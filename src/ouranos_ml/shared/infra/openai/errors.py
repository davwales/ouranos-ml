"""Translation of LLM backend failures (raised by the ``openai`` SDK) into HTTP responses."""

import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


async def _upstream_error(_: Request, exc: Exception) -> JSONResponse:
    """Forward upstream status errors as-is; an unreachable backend is a 502, a timeout a 504."""
    if isinstance(exc, openai.APIStatusError):
        error = exc.body if isinstance(exc.body, dict) else {"message": exc.message, "type": "upstream_error"}
        return JSONResponse(status_code=exc.status_code, content={"error": error})
    status_code = 504 if isinstance(exc, openai.APITimeoutError) else 502
    return JSONResponse(status_code=status_code, content={"error": {"message": str(exc), "type": "server_error"}})


def register_openai_error_handler(app: FastAPI) -> None:
    """Handle ``openai.APIError`` (upstream status, timeout, and connection errors) app-wide."""
    app.add_exception_handler(openai.APIError, _upstream_error)
