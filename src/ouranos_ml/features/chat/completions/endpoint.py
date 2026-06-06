from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ouranos_ml.features.chat.completions.schemas import ChatCompletionsRequest, ChatCompletionsResponse
from ouranos_ml.features.chat.completions.service import handle, handle_stream
from ouranos_ml.shared.utils import done_event, server_side_event


async def _completions(request: ChatCompletionsRequest) -> ChatCompletionsResponse | StreamingResponse:
    """Streams a completion response from the specified chat history and configuration."""
    if request.stream:

        async def stream_events() -> AsyncIterator[str]:
            async for chunk in handle_stream(request):
                yield server_side_event(chunk)
            yield done_event()

        return StreamingResponse(stream_events(), media_type="text/event-stream")
    return await handle(request)


def register(router: APIRouter) -> None:
    """Register chat completions endpoints on the provided router."""
    router.post("/completions", response_model=None)(_completions)