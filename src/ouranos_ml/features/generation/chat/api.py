from fastapi.responses import StreamingResponse

from ouranos_ml.features.generation.chat.handler import handle
from ouranos_ml.features.generation.chat.schemas import ChatCompletionsRequest
from ouranos_ml.features.generation.router import generation_router


@generation_router.post("/chat", deprecated=True)
async def chat(request: ChatCompletionsRequest) -> StreamingResponse:
    """Streams a completion response from the specified chat history and configuration."""
    return StreamingResponse(handle(request), media_type="text/plain")
