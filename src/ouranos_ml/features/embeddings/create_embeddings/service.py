import structlog

from ouranos_ml.features.embeddings.create_embeddings.schemas import (
    CreateEmbeddingsRequest,
    CreateEmbeddingsResponse,
    Embedding,
    Usage,
)
from ouranos_ml.shared.infra.clients.llm_client import get_openai_client

logger = structlog.get_logger(__name__)


def _input_count(input_value: str | list[str]) -> int:
    """Count embedding inputs, treating a bare string as a single input."""
    return len(input_value) if isinstance(input_value, list) else 1


async def handle(request: CreateEmbeddingsRequest) -> CreateEmbeddingsResponse:
    """Creates embeddings for the given input using the LLM service."""
    logger.debug("embeddings requested", model=request.model, input_count=_input_count(request.input))

    try:
        client = get_openai_client()
        response = await client.embeddings.create(model=request.model, input=request.input)
    except Exception:
        logger.error("embeddings upstream failed", model=request.model, exc_info=True)
        raise

    logger.debug("embeddings complete", model=request.model, total_tokens=response.usage.total_tokens)
    return CreateEmbeddingsResponse(
        model=response.model,
        data=[Embedding(index=e.index, embedding=e.embedding) for e in response.data],
        usage=Usage(prompt_tokens=response.usage.prompt_tokens, total_tokens=response.usage.total_tokens),
    )
