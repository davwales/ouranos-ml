from ouranos_ml.features.embeddings.create_embeddings.schemas import (
    CreateEmbeddingsResponse,
    Embedding,
    Usage,
)
from ouranos_ml.shared.infra.clients.llm_client import get_openai_client


async def embed(model: str, input: str | list[str]) -> CreateEmbeddingsResponse:
    """Creates embeddings using the LLM service for the given request."""
    client = get_openai_client()
    response = await client.embeddings.create(model=model, input=input)
    return CreateEmbeddingsResponse(
        model=response.model,
        data=[Embedding(index=e.index, embedding=e.embedding) for e in response.data],
        usage=Usage(prompt_tokens=response.usage.prompt_tokens, total_tokens=response.usage.total_tokens),
    )
