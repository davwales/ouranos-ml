from ouranos_ml.features.models.list_models.schemas import ListModelsResponse
from ouranos_ml.features.models.list_models.service import get_models
from ouranos_ml.features.models.router import models_router


@models_router.get("/")
def list_models() -> ListModelsResponse:
    """Lists all models available in the system."""
    return get_models()
