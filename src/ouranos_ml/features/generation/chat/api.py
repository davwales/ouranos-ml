from fastapi.responses import StreamingResponse

from ouranos_ml.features.chat.completions.api import completions
from ouranos_ml.features.chat.completions.schemas import ChatCompletionsRequest
from ouranos_ml.features.generation.router import generation_router


@generation_router.post("/chat", deprecated=True)
async def chat(request: ChatCompletionsRequest) -> StreamingResponse:
    """Streams a completion response from the specified chat history and configuration."""
    return await completions(request)
