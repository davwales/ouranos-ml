import pytest

from ouranos_ml.features.embeddings.create_embeddings.schemas import (
    CreateEmbeddingsRequest,
    CreateEmbeddingsResponse,
    Embedding,
    Usage,
)


def test_create_embeddings_request_when_input_is_string_should_accept():
    # Arrange
    input_data = {"model": "text-embedding-ada-002", "input": "hello"}

    # Act
    request = CreateEmbeddingsRequest.model_validate(input_data)

    # Assert
    assert request.input == "hello"
    assert request.model == "text-embedding-ada-002"


def test_create_embeddings_request_when_input_is_list_should_accept():
    # Arrange
    input_data = {"model": "text-embedding-ada-002", "input": ["a", "b"]}

    # Act
    request = CreateEmbeddingsRequest.model_validate(input_data)

    # Assert
    assert request.input == ["a", "b"]


def test_create_embeddings_request_when_model_missing_should_raise_validation_error():
    # Arrange
    input_data = {"input": "hello"}

    # Act
    with pytest.raises(ValueError):
        CreateEmbeddingsRequest.model_validate(input_data)


def test_create_embeddings_request_when_serialized_should_use_camel_case():
    # Arrange
    request = CreateEmbeddingsRequest(model="test-model", input="hello")

    # Act
    data = request.model_dump(by_alias=True)

    # Assert
    assert "promptTokens" not in data
    assert data["model"] == "test-model"
    assert data["input"] == "hello"


def test_embedding_when_default_should_have_object_field():
    # Arrange
    embedding = Embedding(index=0, embedding=[0.1, 0.2, 0.3])

    # Act
    data = embedding.model_dump(by_alias=True)

    # Assert
    assert data["object"] == "embedding"


def test_usage_when_created_should_include_prompt_and_total_tokens():
    # Arrange
    usage = Usage(prompt_tokens=10, total_tokens=10)

    # Act
    data = usage.model_dump(by_alias=True)

    # Assert
    assert data == {"promptTokens": 10, "totalTokens": 10}


def test_create_embeddings_response_when_default_should_have_object_list():
    # Arrange
    response = CreateEmbeddingsResponse(
        model="test-model",
        data=[Embedding(index=0, embedding=[0.1, 0.2])],
        usage=Usage(prompt_tokens=1, total_tokens=1),
    )

    # Act
    data = response.model_dump(by_alias=True)

    # Assert
    assert data["object"] == "list"
