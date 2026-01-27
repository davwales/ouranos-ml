from ouranos_ml.features.models.list_models.integration import list_downloaded_models
from ouranos_ml.features.models.list_models.schemas import ListModelsResponse


async def handle() -> ListModelsResponse:
    """Gets all models from the LLM host."""
    return await list_downloaded_models()
