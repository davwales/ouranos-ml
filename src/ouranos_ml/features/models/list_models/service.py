import structlog

from ouranos_ml.features.models.list_models.schemas import ListModelsResponse, Model
from ouranos_ml.shared.infra.clients.llm_client import get_openai_client

logger = structlog.get_logger(__name__)


async def handle() -> ListModelsResponse:
    """Lists all downloaded models from the LLM service."""
    logger.debug("models list requested")

    try:
        client = get_openai_client()
        models = [model async for model in client.models.list()]
    except Exception:
        logger.error("models list upstream failed", exc_info=True)
        raise

    logger.debug("models list complete", model_count=len(models))
    return ListModelsResponse(
        data=[Model(id=model.id, owned_by=model.owned_by, created=model.created) for model in models]
    )
