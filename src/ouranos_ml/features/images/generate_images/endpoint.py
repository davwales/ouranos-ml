from collections.abc import AsyncIterator
from contextlib import aclosing

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ouranos_ml.features.images.generate_images.schemas import GenerateImagesRequest, GenerateImagesResponse
from ouranos_ml.features.images.generate_images.service import handle, open_stream
from ouranos_ml.shared.utils import server_side_event


async def _generate_images(request: GenerateImagesRequest) -> GenerateImagesResponse | StreamingResponse:
    """Generates images, streamed as named SSE progress events (no ``[DONE]``) when ``stream`` is true."""
    if request.stream:
        events = await open_stream(request)

        async def stream_events() -> AsyncIterator[str]:
            async with aclosing(events):
                async for event in events:
                    yield server_side_event(event, event=event.type)

        return StreamingResponse(stream_events(), media_type="text/event-stream")
    return await handle(request)


def register(router: APIRouter) -> None:
    """Register generate-images endpoints on the provided router."""
    router.post("/generations", response_model=None)(_generate_images)
