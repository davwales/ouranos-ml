from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ouranos_ml.features.images.generate_images.schemas import GenerateImagesRequest
from ouranos_ml.features.images.generation import handle, open_stream
from ouranos_ml.features.images.schemas import ImagesResponse
from ouranos_ml.shared.utils import sse_response


async def _generate_images(request: GenerateImagesRequest) -> ImagesResponse | StreamingResponse:
    """Generates images, streamed as named SSE progress events (no ``[DONE]``) when ``stream`` is true."""
    if request.stream:
        return sse_response(await open_stream(request), named=True)
    return await handle(request)


def register(router: APIRouter) -> None:
    """Register generate-images endpoints on the provided router."""
    router.post(
        "/generations",
        response_model=None,
        responses={
            200: {
                "model": ImagesResponse,
                "description": "Generated images as JSON, or a stream of named SSE progress events (no `[DONE]`) when `stream` is true.",
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
            }
        },
    )(_generate_images)
