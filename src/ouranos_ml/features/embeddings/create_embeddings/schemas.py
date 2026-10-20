from typing import Literal

from ouranos_ml.shared.domain.core.base_schema import BaseSchema
from pydantic import Field


class CreateEmbeddingsRequest(BaseSchema):
    """Request to create embeddings."""

    input: str | list[str]
    model: str
    dimensions: int | None = Field(default=None, ge=1)
    encoding_format: Literal["float", "base64"] | None = None


class Embedding(BaseSchema):
    """Response containing the embeddings."""

    object: str = "embedding"
    embedding: list[float] | str  # str when the client asked for encoding_format="base64"
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
