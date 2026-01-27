from ouranos_ml.shared.domain.core.base_schema import BaseSchema


class CreateEmbeddingsRequest(BaseSchema):
    """Request to create embeddings."""

    input: str | list[str]
    model: str


class Embedding(BaseSchema):
    """Response containing the embeddings."""

    object: str = "embedding"
    embedding: list[int | float]
    index: int


class Usage(BaseSchema):
    """Response containing the usage information."""

    prompt_tokens: int
    total_tokens: int


class CreateEmbeddingsResponse(BaseSchema):
    """Wrapper around the response object."""

    object: str = "list"
    model: str
    data: list[Embedding]
    usage: Usage
