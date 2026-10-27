from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ouranos_ml.features.chat.completions.schemas import ChatCompletionsRequest, ChatCompletionsResponse
from ouranos_ml.features.chat.completions.service import handle, open_stream
from ouranos_ml.shared.utils import sse_response


async def _completions(request: ChatCompletionsRequest) -> ChatCompletionsResponse | StreamingResponse:
    """Streams a completion response from the specified chat history and configuration."""
    if request.stream:
        return sse_response(await open_stream(request), done=True)
    return await handle(request)


def register(router: APIRouter) -> None:
    """Register chat completions endpoints on the provided router."""
    router.post(
        "/completions",
        response_model=None,
        responses={
            200: {
                "model": ChatCompletionsResponse,
                "description": "A chat completion as JSON, or an SSE stream of chunks ending in `[DONE]` when `stream` is true.",
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
            }
        },
    )(_completions)
