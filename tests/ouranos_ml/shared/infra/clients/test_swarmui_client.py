import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from websockets.exceptions import ConnectionClosedError, InvalidStatus

from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.infra.clients.swarmui_client import SwarmUIClient, get_swarmui_client
from ouranos_ml.shared.infra.swarmui.errors import (
    SwarmUIError,
    SwarmUINotConfiguredError,
    SwarmUITimeoutError,
    SwarmUIUnavailableError,
)
from tests.ouranos_ml.shared.openai_mocks import AsyncIteratorMock

CONNECT = "ouranos_ml.shared.infra.clients.swarmui_client.connect"
BASE_URL = "http://swarm:7801"


def _client(handler, *, token: str = "") -> SwarmUIClient:
    http = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    return SwarmUIClient(BASE_URL, token=token, http=http)


async def _frames(raw: list[str], error: Exception):
    for frame in raw:
        yield frame
    raise error


def _socket(messages: list[dict], *, error: Exception | None = None) -> MagicMock:
    raw = [json.dumps(message) for message in messages]
    socket = MagicMock()
    socket.send = AsyncMock()
    socket.close = AsyncMock()
    socket.__aiter__ = MagicMock(return_value=AsyncIteratorMock(raw) if error is None else _frames(raw, error))
    return socket


@pytest.mark.asyncio
async def test_new_session_when_called_should_return_session_id():
    # Arrange
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"session_id": "abc"})

    client = _client(handler)

    # Act
    session_id = await client.new_session()

    # Assert
    assert session_id == "abc"
    assert requests[0].url.path == "/API/GetNewSession"
    assert json.loads(requests[0].content) == {}


@pytest.mark.asyncio
async def test_call_when_called_should_post_session_and_body():
    # Arrange
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"model": {"name": "m"}})

    client = _client(handler)

    # Act
    response = await client.call("DescribeModel", "s1", modelName="m")

    # Assert
    assert response == {"model": {"name": "m"}}
    assert bodies == [{"session_id": "s1", "modelName": "m"}]


@pytest.mark.asyncio
async def test_call_when_swarmui_replies_with_error_should_raise_with_its_message():
    # Arrange
    client = _client(lambda _: httpx.Response(200, json={"error": "Model not found."}))

    # Act & Assert
    with pytest.raises(SwarmUIError, match="Model not found."):
        await client.call("DescribeModel", "s1", modelName="nope")


@pytest.mark.asyncio
async def test_call_when_status_error_without_body_should_raise():
    # Arrange
    client = _client(lambda _: httpx.Response(500, text="boom"))

    # Act & Assert
    with pytest.raises(SwarmUIError, match="500"):
        await client.call("ListModels", "s1")


@pytest.mark.asyncio
async def test_call_when_unreachable_should_raise_unavailable():
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = _client(handler)

    # Act & Assert
    with pytest.raises(SwarmUIUnavailableError):
        await client.call("ListModels", "s1")


@pytest.mark.asyncio
async def test_call_when_timed_out_should_raise_timeout():
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = _client(handler)

    # Act & Assert
    with pytest.raises(SwarmUITimeoutError):
        await client.call("ListModels", "s1")


@pytest.mark.asyncio
async def test_interrupt_when_called_should_interrupt_only_this_session():
    # Arrange
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"success": True})

    client = _client(handler)

    # Act
    await client.interrupt("s1")

    # Assert
    assert requests[0].url.path == "/API/InterruptAll"
    assert json.loads(requests[0].content) == {"session_id": "s1", "other_sessions": False}


@pytest.mark.asyncio
async def test_fetch_file_when_found_should_return_bytes():
    # Arrange
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, content=b"png-bytes")

    client = _client(handler)

    # Act
    content = await client.fetch_file("View/local/raw/a fox.png")

    # Assert
    assert content == b"png-bytes"
    assert paths == ["/View/local/raw/a fox.png"]


@pytest.mark.asyncio
async def test_fetch_file_when_missing_should_raise():
    # Arrange
    client = _client(lambda _: httpx.Response(404))

    # Act & Assert
    with pytest.raises(SwarmUIError, match="404"):
        await client.fetch_file("View/missing.png")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [(httpx.ConnectError("refused"), SwarmUIUnavailableError), (httpx.ReadTimeout("slow"), SwarmUITimeoutError)],
)
async def test_fetch_file_when_transport_fails_should_raise(error, expected):
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:
        error.request = request
        raise error

    client = _client(handler)

    # Act & Assert
    with pytest.raises(expected):
        await client.fetch_file("View/a.png")


@pytest.mark.asyncio
async def test_open_generation_when_connected_should_send_payload_and_yield_until_close_intention():
    # Arrange
    socket = _socket([{"status": {}}, {"image": "data:,"}, {"socket_intention": "close"}, {"status": {}}])
    client = SwarmUIClient(BASE_URL, token="tok")

    # Act
    with patch(CONNECT, AsyncMock(return_value=socket)) as connect:
        messages = await client.open_generation("s1", {"prompt": "fox", "images": 1})
        received = [message async for message in messages]

    # Assert
    assert connect.await_args.args[0] == "ws://swarm:7801/API/GenerateText2ImageWS"
    assert connect.await_args.kwargs["additional_headers"] == {"Cookie": "swarm_token=tok"}
    assert json.loads(socket.send.await_args.args[0]) == {"session_id": "s1", "prompt": "fox", "images": 1}
    assert received == [{"status": {}}, {"image": "data:,"}]
    socket.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_open_generation_when_unreachable_should_raise_unavailable_eagerly():
    # Arrange
    client = SwarmUIClient(BASE_URL)

    # Act & Assert
    with patch(CONNECT, AsyncMock(side_effect=OSError("refused"))), pytest.raises(SwarmUIUnavailableError):
        await client.open_generation("s1", {})


@pytest.mark.asyncio
async def test_open_generation_when_handshake_rejected_should_raise_unavailable():
    # Arrange
    client = SwarmUIClient(BASE_URL)
    rejection = InvalidStatus(MagicMock(status_code=401))

    # Act & Assert
    with patch(CONNECT, AsyncMock(side_effect=rejection)), pytest.raises(SwarmUIUnavailableError):
        await client.open_generation("s1", {})


@pytest.mark.asyncio
async def test_open_generation_when_connection_drops_should_raise_unavailable_and_close():
    # Arrange
    socket = _socket([{"status": {}}], error=ConnectionClosedError(None, None))
    client = SwarmUIClient(BASE_URL)

    # Act
    with patch(CONNECT, AsyncMock(return_value=socket)):
        messages = await client.open_generation("s1", {})
        first = await anext(messages)
        with pytest.raises(SwarmUIUnavailableError):
            await anext(messages)

    # Assert
    assert first == {"status": {}}
    socket.close.assert_awaited_once()


def test_get_swarmui_client_when_base_url_unset_should_raise_not_configured():
    # Arrange
    get_swarmui_client.cache_clear()
    settings = MagicMock(swarmui_base_url="")

    # Act & Assert
    with (
        patch("ouranos_ml.shared.infra.clients.swarmui_client.get_settings", return_value=settings),
        pytest.raises(SwarmUINotConfiguredError),
    ):
        get_swarmui_client()


def test_get_swarmui_client_when_configured_should_return_cached_client():
    # Arrange
    get_swarmui_client.cache_clear()
    settings = MagicMock(swarmui_base_url=BASE_URL, swarmui_token="")

    # Act
    with patch("ouranos_ml.shared.infra.clients.swarmui_client.get_settings", return_value=settings):
        first = get_swarmui_client()
        second = get_swarmui_client()
    get_swarmui_client.cache_clear()
    get_settings.cache_clear()

    # Assert
    assert isinstance(first, SwarmUIClient)
    assert first is second
