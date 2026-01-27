from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse

from ouranos_ml.features.chat.completions.handler import handle, handle_stream
from ouranos_ml.features.chat.completions.schemas import ChatCompletionsRequest, ChatCompletionsResponse
from ouranos_ml.features.chat.router import chat_router
from ouranos_ml.shared.utils import done_event, server_side_event


@chat_router.post("/completions", response_model=None)
async def completions(request: ChatCompletionsRequest) -> ChatCompletionsResponse | StreamingResponse:
    """Streams a completion response from the specified chat history and configuration."""
    if request.stream:

        async def stream_events() -> AsyncIterator[str]:
            async for chunk in handle_stream(request):
                yield server_side_event(chunk)
            yield done_event()

        return StreamingResponse(stream_events(), media_type="text/event-stream")
    return await handle(request)
