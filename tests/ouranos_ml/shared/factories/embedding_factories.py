"""Factory functions for creating embeddings test data."""

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
) -> CreateEmbeddingsRequest:
    """Create a CreateEmbeddingsRequest with sensible defaults."""
    return CreateEmbeddingsRequest(model=model, input=input)


def make_embedding(
    *,
    index: int = 0,
    embedding: list[int | float] | None = None,
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
        data=embeddings or [make_embedding()],
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens),
    )
