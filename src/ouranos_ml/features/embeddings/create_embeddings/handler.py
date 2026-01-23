from collections.abc import Sequence

from ouranos_ml.features.embeddings.create_embeddings.integration import count_tokens, embed
from ouranos_ml.features.embeddings.create_embeddings.schemas import (
    CreateEmbeddingsRequest,
    CreateEmbeddingsResponse,
    EmbeddingResponse,
    UsageResponse,
)


async def handle(request: CreateEmbeddingsRequest) -> CreateEmbeddingsResponse:
    """Creates embeddings."""
    token_counts = await count_tokens(request.model, request.input)
    embeddings = await embed(request.model, request.input)

    if not isinstance(embeddings[0], Sequence):
        embeddings = [embeddings]

    return CreateEmbeddingsResponse(
        model=request.model,
        data=[EmbeddingResponse(embedding=value, index=index) for (index, value) in enumerate(embeddings)],
        usage=UsageResponse(prompt_tokens=sum(token_counts), total_tokens=sum(token_counts)),
    )
