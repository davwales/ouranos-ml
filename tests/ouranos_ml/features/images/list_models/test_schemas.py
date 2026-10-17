from ouranos_ml.features.images.list_models.schemas import ImageModel, ListImageModelsResponse


def test_list_image_models_response_when_serialized_should_use_snake_case():
    # Arrange
    response = ListImageModelsResponse(
        default_model="flux-2-klein-9b",
        data=[ImageModel(id="flux-2-klein-9b", default_size="1024x1024", default_steps=4, default_cfg_scale=1.0)],
    )

    # Act
    body = response.model_dump(by_alias=True)

    # Assert
    assert body["object"] == "list"
    assert body["default_model"] == "flux-2-klein-9b"
    assert body["data"][0]["default_cfg_scale"] == 1.0
    assert body["data"][0]["loaded"] is False


def test_image_model_when_camel_case_input_should_populate_fields():
    # Arrange
    payload = {"id": "m", "defaultSize": "512x512", "usageHint": "hint"}

    # Act
    model = ImageModel.model_validate(payload)

    # Assert
    assert model.default_size == "512x512"
    assert model.usage_hint == "hint"
