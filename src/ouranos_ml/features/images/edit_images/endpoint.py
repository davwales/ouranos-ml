from collections.abc import AsyncIterator
from contextlib import aclosing

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ouranos_ml.features.images.edit_images.schemas import EditImagesRequest
from ouranos_ml.features.images.edit_images.service import handle, open_stream
from ouranos_ml.features.images.schemas import ImagesResponse
from ouranos_ml.shared.utils import server_side_event


async def _edit_images(request: EditImagesRequest) -> ImagesResponse | StreamingResponse:
    """Edits an image, streamed as named SSE progress events (no ``[DONE]``) when ``stream`` is true."""
    if request.stream:
        events = await open_stream(request)

        async def stream_events() -> AsyncIterator[str]:
            async with aclosing(events):
                async for event in events:
                    yield server_side_event(event, event=event.type)

        return StreamingResponse(stream_events(), media_type="text/event-stream")
    return await handle(request)


def register(router: APIRouter) -> None:
    """Register edit-images endpoints on the provided router."""
    router.post(
        "/edits",
        response_model=None,
        responses={
            200: {
                "model": ImagesResponse,
                "description": "Edited images as JSON, or a stream of named SSE progress events (no `[DONE]`) when `stream` is true. The events are the same as `/images/generations`.",
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
            }
        },
    )(_edit_images)
