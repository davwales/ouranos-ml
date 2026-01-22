from collections.abc import Sequence

from ouranos_ml.features.embeddings.create_embeddings.integration import count_tokens, embed
from ouranos_ml.features.embeddings.create_embeddings.schemas import (
    CreateEmbeddingsRequest,
    CreateEmbeddingsResponse,
    EmbeddingResponse,
    UsageResponse,
)


def get_embeddings(request: CreateEmbeddingsRequest) -> CreateEmbeddingsResponse:
    """Creates embeddings."""
    token_counts = count_tokens(request.model, request.input)
    embeddings = embed(request.model, request.input)

    if not isinstance(embeddings[0], Sequence):
        embeddings = [embeddings]

    return CreateEmbeddingsResponse(
        model=request.model,
        data=[EmbeddingResponse(embedding=value, index=index) for (index, value) in enumerate(embeddings)],
        usage=UsageResponse(prompt_tokens=sum(token_counts), total_tokens=sum(token_counts)),
    )
