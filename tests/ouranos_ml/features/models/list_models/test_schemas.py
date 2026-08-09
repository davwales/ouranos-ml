import pytest

from ouranos_ml.features.models.list_models.schemas import ListModelsResponse, Model


def test_model_when_default_should_have_object_field():
    # Arrange
    model = Model(id="test-model", created=1700000000)

    # Act
    result = model.object

    # Assert
    assert result == "model"


def test_model_when_default_should_have_owned_by_ouranos_ml():
    # Arrange
    model = Model(id="test-model", created=1700000000)

    # Act
    result = model.owned_by

    # Assert
    assert result == "ouranos-ml"


def test_model_when_serialized_should_use_camel_case():
    # Arrange
    model = Model(id="test-model", created=1700000000, owned_by="org-a")

    # Act
    data = model.model_dump(by_alias=True)

    # Assert
    assert "ownedBy" in data
    assert data["ownedBy"] == "org-a"


def test_model_when_id_missing_should_raise_validation_error():
    # Arrange
    input_data = {"created": 1700000000}

    # Act
    with pytest.raises(ValueError):
        Model.model_validate(input_data)


def test_list_models_response_when_default_should_have_object_list():
    # Arrange
    response = ListModelsResponse(data=[])

    # Act
    result = response.object

    # Assert
    assert result == "list"


def test_list_models_response_when_empty_data_should_serialize():
    # Arrange
    response = ListModelsResponse(data=[])

    # Act
    data = response.model_dump(by_alias=True)

    # Assert
    assert data["data"] == []
    assert data["object"] == "list"
