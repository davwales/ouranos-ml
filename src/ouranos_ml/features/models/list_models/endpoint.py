from fastapi import APIRouter

from ouranos_ml.features.models.list_models.schemas import ListModelsResponse
from ouranos_ml.features.models.list_models.service import handle


async def _list_models() -> ListModelsResponse:
    """Lists all models available in the system."""
    return await handle()


def register(router: APIRouter) -> None:
    """Register list-models endpoints on the provided router."""
    router.get("")(_list_models)