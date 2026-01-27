from ouranos_ml.features.embeddings.create_embeddings.integration import embed
from ouranos_ml.features.embeddings.create_embeddings.schemas import (
    CreateEmbeddingsRequest,
    CreateEmbeddingsResponse,
)


async def handle(request: CreateEmbeddingsRequest) -> CreateEmbeddingsResponse:
    """Creates embeddings."""
    return await embed(request.model, request.input)
