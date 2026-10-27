from collections.abc import AsyncGenerator
from typing import Any, Literal, Protocol, overload

from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from pydantic import BaseModel


def server_side_event[T: BaseModel](data: T, event: str | None = None) -> str:
    """Helper function to create server-side events."""
    parts: list[str] = []
    if event:
        parts.append(f"event: {event}")
    parts.append(f"data: {data.model_dump_json(by_alias=True)}")
    return "\n".join(parts) + "\n\n"


def done_event() -> str:
    """Helper function to create a server-side event indicating completion."""
    return "data: [DONE]\n\n"


class _NamedEvent(Protocol):
    """An event model whose ``type`` names its SSE event."""

    @property
    def type(self) -> str: ...

    def model_dump_json(self, *, by_alias: bool | None = None) -> str: ...


@overload
def sse_response(
    events: AsyncGenerator[_NamedEvent], *, named: Literal[True], done: bool = False
) -> StreamingResponse: ...


@overload
def sse_response(
    events: AsyncGenerator[BaseModel], *, named: Literal[False] = False, done: bool = False
) -> StreamingResponse: ...


def sse_response(events: AsyncGenerator[Any], *, named: bool = False, done: bool = False) -> StreamingResponse:
    """Stream ``events`` as an SSE response, closing the generator when the response ends.

    Starlette never closes the body iterator, so without this a stream cut short by a client disconnect would leave
    ``events`` (and the upstream stream or SwarmUI generation it holds) to garbage collection. Closing it here runs its
    cleanup as soon as the response finishes, however it finishes.

    Args:
        events: The event models to send, in order.
        named: Send each event as ``event: <event.type>``, like ``/responses``. Otherwise events are unnamed ``data:``
            chunks, like ``/chat/completions``.
        done: Send ``data: [DONE]`` after the last event, if the stream completes.
    """
    return _SSEResponse(events, named=named, done=done)


class _SSEResponse(StreamingResponse):
    """A ``text/event-stream`` response that closes its event generator once sent."""

    def __init__(self, events: AsyncGenerator[Any], *, named: bool, done: bool) -> None:
        self._events = events
        super().__init__(self._serialize(named, done), media_type="text/event-stream")

    async def _serialize(self, named: bool, done: bool) -> AsyncGenerator[str]:
        async for event in self._events:
            yield server_side_event(event, event=event.type if named else None)
        if done:
            yield done_event()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._events.aclose()
