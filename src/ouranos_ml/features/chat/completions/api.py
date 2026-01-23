from fastapi.responses import StreamingResponse

from ouranos_ml.features.chat.completions.handler import handle
from ouranos_ml.features.chat.completions.schemas import ChatCompletionsRequest
from ouranos_ml.features.chat.router import chat_router


@chat_router.post("/completions")
def completions(request: ChatCompletionsRequest) -> StreamingResponse:
    """Streams a completion response from the specified chat history and configuration."""
    return StreamingResponse(handle(request), media_type="text/plain")
