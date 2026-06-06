from ouranos_ml.features.embeddings.create_embeddings.schemas import (
    CreateEmbeddingsRequest,
    CreateEmbeddingsResponse,
    Embedding,
    Usage,
)
from ouranos_ml.shared.infra.clients.llm_client import get_openai_client


async def handle(request: CreateEmbeddingsRequest) -> CreateEmbeddingsResponse:
    """Creates embeddings for the given input using the LLM service."""
    client = get_openai_client()
    response = await client.embeddings.create(model=request.model, input=request.input)
    return CreateEmbeddingsResponse(
        model=response.model,
        data=[Embedding(index=e.index, embedding=e.embedding) for e in response.data],
        usage=Usage(prompt_tokens=response.usage.prompt_tokens, total_tokens=response.usage.total_tokens),
    )
