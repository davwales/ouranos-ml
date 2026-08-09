from datetime import UTC, datetime

from ouranos_ml.features.models.list_models.schemas import ListModelsResponse, Model
from ouranos_ml.shared.infra.clients.llm_client import get_openai_client


async def handle() -> ListModelsResponse:
    """Lists all downloaded models from the LLM service."""
    client = get_openai_client()
    return ListModelsResponse(
        data=[
            Model(id=model.id, owned_by=model.owned_by, created=int(datetime.now(UTC).timestamp()))
            async for model in client.models.list()
        ]
    )
