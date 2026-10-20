"""Factory functions for creating embeddings test data."""

from typing import Literal
from unittest.mock import AsyncMock, MagicMock

from ouranos_ml.features.embeddings.create_embeddings.schemas import (
    CreateEmbeddingsRequest,
    CreateEmbeddingsResponse,
    Embedding,
    Usage,
)


def make_embedding_request(
    *,
    model: str = "test-embedding-model",
    input: str | list[str] = "hello world",
    dimensions: int | None = None,
    encoding_format: Literal["float", "base64"] | None = None,
) -> CreateEmbeddingsRequest:
    """Create a CreateEmbeddingsRequest with sensible defaults."""
    return CreateEmbeddingsRequest(model=model, input=input, dimensions=dimensions, encoding_format=encoding_format)


def make_embedding(
    *,
    index: int = 0,
    embedding: list[float] | str | None = None,
) -> Embedding:
    """Create an Embedding with sensible defaults."""
    return Embedding(
        embedding=embedding if embedding is not None else [0.1, 0.2, 0.3],
        index=index,
    )


def make_embedding_response(
    *,
    model: str = "test-embedding-model",
    embeddings: list[Embedding] | None = None,
    prompt_tokens: int = 5,
    total_tokens: int = 5,
) -> CreateEmbeddingsResponse:
    """Create a CreateEmbeddingsResponse with sensible defaults."""
    return CreateEmbeddingsResponse(
        model=model,
        data=embeddings if embeddings is not None else [make_embedding()],
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens),
    )


def make_mock_embeddings_client(*, embedding: list[float] | str, model: str = "test-model") -> MagicMock:
    """Create a mock OpenAI client whose ``embeddings.create`` returns a single upstream embedding."""
    mock_embedding_data = MagicMock()
    mock_embedding_data.index = 0
    mock_embedding_data.embedding = embedding
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 1
    mock_usage.total_tokens = 1
    mock_response = MagicMock()
    mock_response.model = model
    mock_response.data = [mock_embedding_data]
    mock_response.usage = mock_usage
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    return mock_client
