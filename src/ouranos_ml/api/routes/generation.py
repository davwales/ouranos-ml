from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ouranos_ml.application.queries.text.generate_chat import generate_chat
from ouranos_ml.application.queries.text.requests import ChatGenerationRequest

router = APIRouter(prefix="/generation")


@router.post("/chat")
def chat(request: ChatGenerationRequest) -> StreamingResponse:
    """Streams a completion response from the specified chat history and configuration."""
    return StreamingResponse(generate_chat(request), media_type="text/plain")
