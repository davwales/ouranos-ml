from datetime import datetime

from dateutil.tz import UTC

from ouranos_ml.features.models.list_models.integration import list_downloaded_models
from ouranos_ml.features.models.list_models.schemas import ListModelsResponse, ModelResponse


def get_models() -> ListModelsResponse:
    """Gets all models from the LLM host."""
    downloaded_models = list_downloaded_models()
    return ListModelsResponse(
        data=[
            ModelResponse(id=model_key, created=int(datetime.now(UTC).timestamp())) for model_key in downloaded_models
        ]
    )
