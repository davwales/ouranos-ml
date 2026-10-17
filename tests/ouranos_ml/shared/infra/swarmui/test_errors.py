from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from ouranos_ml.shared.infra.swarmui.errors import (
    SwarmUIError,
    SwarmUIModelNotFoundError,
    SwarmUINotConfiguredError,
    SwarmUITimeoutError,
    SwarmUIUnavailableError,
    register_swarmui_error_handler,
)

_ERRORS = {
    "generic": SwarmUIError("Invalid value for parameter Sampler"),
    "not-configured": SwarmUINotConfiguredError("not configured"),
    "unavailable": SwarmUIUnavailableError("unreachable"),
    "not-found": SwarmUIModelNotFoundError("Model 'x' not found"),
    "timeout": SwarmUITimeoutError("too slow"),
}


def _build_app() -> FastAPI:
    application = FastAPI()
    register_swarmui_error_handler(application)

    @application.post("/fail/{kind}")
    async def _fail(kind: str) -> dict:
        raise _ERRORS[kind]

    return application


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=_build_app(), raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "status_code", "error_type"),
    [
        ("generic", 502, "upstream_error"),
        ("not-configured", 503, "not_configured"),
        ("unavailable", 502, "server_error"),
        ("not-found", 404, "model_not_found"),
        ("timeout", 504, "timeout"),
    ],
)
async def test_swarmui_error_when_raised_should_map_to_status_and_typed_body(client, kind, status_code, error_type):
    # Act
    response = await client.post(f"/fail/{kind}")

    # Assert
    assert response.status_code == status_code
    assert response.json() == {"error": {"message": _ERRORS[kind].message, "type": error_type}}
