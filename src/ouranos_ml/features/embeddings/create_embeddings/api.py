from ouranos_ml.features.embeddings.create_embeddings.handler import handle
from ouranos_ml.features.embeddings.create_embeddings.schemas import CreateEmbeddingsRequest, CreateEmbeddingsResponse
from ouranos_ml.features.embeddings.router import embeddings_router


@embeddings_router.post("")
async def create_embeddings(request: CreateEmbeddingsRequest) -> CreateEmbeddingsResponse:
    """Create embeddings for the given input."""
    return await handle(request)
