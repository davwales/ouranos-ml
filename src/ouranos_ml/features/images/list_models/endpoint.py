from fastapi import APIRouter

from ouranos_ml.features.images.list_models.schemas import ListImageModelsResponse
from ouranos_ml.features.images.list_models.service import handle


async def _list_models() -> ListImageModelsResponse:
    """Lists the image models available for generation."""
    return await handle()


def register(router: APIRouter) -> None:
    """Register list-image-models endpoints on the provided router."""
    router.get("/models")(_list_models)
