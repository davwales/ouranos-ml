from ouranos_ml.features.models.list_models.handler import handle
from ouranos_ml.features.models.list_models.schemas import ListModelsResponse
from ouranos_ml.features.models.router import models_router


@models_router.get("/")
async def list_models() -> ListModelsResponse:
    """Lists all models available in the system."""
    return await handle()
