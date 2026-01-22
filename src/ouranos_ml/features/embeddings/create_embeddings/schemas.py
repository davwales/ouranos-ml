from ouranos_ml.shared.domain.core.base_schema import BaseSchema


class CreateEmbeddingsRequest(BaseSchema):
    """Request to create embeddings."""

    input: str | list[str]
    model: str


class UsageResponse(BaseSchema):
    """Usage statistics from serving the request."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseSchema):
    """Response containing the embeddings."""

    object: str = "embedding"
    embedding: list[float]
    index: int


class CreateEmbeddingsResponse(BaseSchema):
    """Wrapper around the response object."""

    object: str = "list"
    model: str
    data: list[EmbeddingResponse]
    usage: UsageResponse
