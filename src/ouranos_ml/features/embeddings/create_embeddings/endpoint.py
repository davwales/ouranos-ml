from fastapi import APIRouter

from ouranos_ml.features.embeddings.create_embeddings.schemas import CreateEmbeddingsRequest, CreateEmbeddingsResponse
from ouranos_ml.features.embeddings.create_embeddings.service import handle


async def _create_embeddings(request: CreateEmbeddingsRequest) -> CreateEmbeddingsResponse:
    """Create embeddings for the given input."""
    return await handle(request)


def register(router: APIRouter) -> None:
    """Register create-embeddings endpoints on the provided router."""
    router.post("")(_create_embeddings)