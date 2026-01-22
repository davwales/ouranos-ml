from ouranos_ml.features.embeddings.create_embeddings.schemas import CreateEmbeddingsRequest, CreateEmbeddingsResponse
from ouranos_ml.features.embeddings.create_embeddings.service import get_embeddings
from ouranos_ml.features.embeddings.router import embeddings_router


@embeddings_router.post("/")
def create_embeddings(request: CreateEmbeddingsRequest) -> CreateEmbeddingsResponse:
    """Create embeddings for the given input."""
    return get_embeddings(request)
