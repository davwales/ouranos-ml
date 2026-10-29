from collections.abc import AsyncGenerator
from typing import Any, Literal

import pytest
from starlette.requests import ClientDisconnect

from ouranos_ml.shared.utils import done_event, server_side_event, sse_response
from pydantic import BaseModel, Field


class SimpleModel(BaseModel):
    """Test model with a single field."""

    message: str


class NestedInner(BaseModel):
    """Test inner model for nesting tests."""

    inner_value: int


class NestedOuter(BaseModel):
    """Test outer model that nests another model."""

    outer_name: str
    inner: NestedInner


def test_server_side_event_when_no_event_name_provided_should_return_data_line_only() -> None:
    """Test SSE formatting without event name."""
    # Arrange
    model = SimpleModel(message="hello")

    # Act
    result = server_side_event(model)

    # Assert
    expected = f"data: {model.model_dump_json()}\n\n"
    assert result == expected
    assert "event:" not in result


def test_server_side_event_when_event_name_provided_should_include_event_line() -> None:
    """Test SSE formatting with event name."""
    # Arrange
    model = SimpleModel(message="hello")

    # Act
    result = server_side_event(model, event="update")

    # Assert
    expected = f"event: update\ndata: {model.model_dump_json()}\n\n"
    assert result == expected


def test_server_side_event_when_complex_model_should_dump_json_correctly() -> None:
    """Test SSE formatting with nested model."""
    # Arrange
    inner = NestedInner(inner_value=42)
    model = NestedOuter(outer_name="test", inner=inner)

    # Act
    result = server_side_event(model)

    # Assert
    expected = f"data: {model.model_dump_json()}\n\n"
    assert result == expected
    assert '"outer_name":"test"' in result


def test_done_event_when_called_should_return_done_string() -> None:
    """Test done event returns correct SSE termination string."""
    # Arrange

    # Act
    result = done_event()

    # Assert
    assert result == "data: [DONE]\n\n"


class AliasedModel(BaseModel):
    """Test model whose field has an explicit wire-name alias."""

    json_schema: dict = Field(alias="schema")


def test_server_side_event_when_field_has_alias_should_emit_wire_name() -> None:
    """Test SSE formatting serializes explicit aliases, matching FastAPI's JSON responses."""
    # Arrange
    model = AliasedModel.model_validate({"schema": {"type": "object"}})

    # Act
    result = server_side_event(model)

    # Assert
    assert result == 'data: {"schema":{"type":"object"}}\n\n'


class NamedEvent(BaseModel):
    """Test event model whose ``type`` names its SSE event."""

    type: Literal["thing.happened"] = "thing.happened"
    count: int


async def _events(items: list[NamedEvent], closed: list[bool], error: Exception | None = None) -> AsyncGenerator[Any]:
    """Yield ``items`` (then raise ``error``), recording in ``closed`` when the generator's cleanup runs."""
    try:
        for item in items:
            yield item
        if error is not None:
            raise error
    finally:
        closed.append(True)


async def _send_response(response: Any, bodies: list[bytes], fail_after: int | None = None) -> None:
    """Run ``response`` as an ASGI app, collecting body chunks; after ``fail_after`` chunks, fail like a gone client."""

    async def receive() -> dict[str, Any]:
        return {"type": "http.request"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] != "http.response.body":
            return
        if fail_after is not None and len(bodies) == fail_after:
            raise OSError("client disconnected")
        bodies.append(message["body"])

    await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)


@pytest.mark.asyncio
async def test_sse_response_when_unnamed_with_done_should_send_data_chunks_then_done() -> None:
    """Test unnamed events are sent as data-only chunks followed by ``[DONE]``."""
    # Arrange
    closed: list[bool] = []
    response = sse_response(_events([NamedEvent(count=1), NamedEvent(count=2)], closed), done=True)
    bodies: list[bytes] = []

    # Act
    await _send_response(response, bodies)

    # Assert
    assert response.media_type == "text/event-stream"
    assert b"".join(bodies) == (
        b'data: {"type":"thing.happened","count":1}\n\ndata: {"type":"thing.happened","count":2}\n\ndata: [DONE]\n\n'
    )


@pytest.mark.asyncio
async def test_sse_response_when_named_should_send_event_lines_without_done() -> None:
    """Test named events carry ``event: <type>`` and the stream has no ``[DONE]``."""
    # Arrange
    closed: list[bool] = []
    response = sse_response(_events([NamedEvent(count=1)], closed), named=True)
    bodies: list[bytes] = []

    # Act
    await _send_response(response, bodies)

    # Assert
    assert b"".join(bodies) == b'event: thing.happened\ndata: {"type":"thing.happened","count":1}\n\n'
    assert closed == [True]


@pytest.mark.asyncio
async def test_sse_response_when_client_disconnects_should_close_events() -> None:
    """Test the events generator is closed as soon as the response ends, not left for garbage collection."""
    # Arrange
    closed: list[bool] = []
    events = _events([NamedEvent(count=1), NamedEvent(count=2), NamedEvent(count=3)], closed)
    response = sse_response(events, named=True)
    bodies: list[bytes] = []

    # Act
    with pytest.raises(ClientDisconnect):
        await _send_response(response, bodies, fail_after=1)

    # Assert
    assert len(bodies) == 1
    assert closed == [True]


@pytest.mark.asyncio
async def test_sse_response_when_events_raise_should_propagate_without_done() -> None:
    """Test a mid-stream failure ends the stream without ``[DONE]``, so clients can tell it was cut short."""
    # Arrange
    closed: list[bool] = []
    response = sse_response(_events([NamedEvent(count=1)], closed, error=RuntimeError("upstream broke")), done=True)
    bodies: list[bytes] = []

    # Act & Assert
    with pytest.raises(RuntimeError, match="upstream broke"):
        await _send_response(response, bodies)
    assert b"[DONE]" not in b"".join(bodies)
    assert closed == [True]
