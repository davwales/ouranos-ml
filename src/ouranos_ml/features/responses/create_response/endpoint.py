from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ouranos_ml.features.responses.create_response.schemas import CreateResponseRequest, ResponseObject
from ouranos_ml.features.responses.create_response.service import handle, open_stream
from ouranos_ml.shared.utils import server_side_event


async def _create_response(request: CreateResponseRequest) -> ResponseObject | StreamingResponse:
    """Creates a model response, streamed as named SSE events (no ``[DONE]``) when ``stream`` is true."""
    if request.stream:
        events = await open_stream(request)

        async def stream_events() -> AsyncIterator[str]:
            async for event in events:
                yield server_side_event(event, event=event.type)

        return StreamingResponse(stream_events(), media_type="text/event-stream")
    return await handle(request)


def register(router: APIRouter) -> None:
    """Register create-response endpoints on the provided router."""
    router.post("", response_model=None)(_create_response)
