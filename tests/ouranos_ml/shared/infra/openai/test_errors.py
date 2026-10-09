from collections.abc import AsyncGenerator

import httpx
import openai
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from ouranos_ml.shared.infra.openai.errors import register_openai_error_handler

_UPSTREAM_REQUEST = httpx.Request("POST", "http://upstream/v1/responses")


def _build_app() -> FastAPI:
    application = FastAPI()
    register_openai_error_handler(application)

    @application.post("/upstream/{status_code}")
    async def _upstream(status_code: int, dict_body: bool = True) -> dict:
        response = httpx.Response(status_code, request=_UPSTREAM_REQUEST)
        body = (
            {"message": "upstream says no", "type": "upstream_type", "param": "p", "code": "c"} if dict_body else None
        )
        raise openai.APIStatusError("upstream says no", response=response, body=body)

    @application.post("/timeout")
    async def _timeout() -> dict:
        raise openai.APITimeoutError(request=_UPSTREAM_REQUEST)

    @application.post("/unreachable")
    async def _unreachable() -> dict:
        raise openai.APIConnectionError(request=_UPSTREAM_REQUEST)

    @application.post("/crash")
    async def _crash() -> dict:
        raise RuntimeError("boom")

    return application


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=_build_app(), raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


@pytest.mark.asyncio
async def test_upstream_error_when_status_error_should_forward_status_and_body(client):
    # Act
    response = await client.post("/upstream/429")

    # Assert
    assert response.status_code == 429
    assert response.json() == {
        "error": {"message": "upstream says no", "type": "upstream_type", "param": "p", "code": "c"}
    }


@pytest.mark.asyncio
async def test_upstream_error_when_body_not_dict_should_wrap_message(client):
    # Act
    response = await client.post("/upstream/503", params={"dict_body": "false"})

    # Assert
    assert response.status_code == 503
    assert response.json() == {"error": {"message": "upstream says no", "type": "upstream_error"}}


@pytest.mark.asyncio
async def test_upstream_error_when_timeout_should_return_504(client):
    # Act
    response = await client.post("/timeout")

    # Assert
    assert response.status_code == 504
    assert response.json()["error"]["type"] == "server_error"


@pytest.mark.asyncio
async def test_upstream_error_when_unreachable_should_return_502(client):
    # Act
    response = await client.post("/unreachable")

    # Assert
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "server_error"


@pytest.mark.asyncio
async def test_upstream_error_when_unrelated_exception_should_propagate_as_500(client):
    # Act
    response = await client.post("/crash")

    # Assert
    assert response.status_code == 500
