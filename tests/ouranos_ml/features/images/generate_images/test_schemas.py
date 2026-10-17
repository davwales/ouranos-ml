import pytest

from ouranos_ml.features.images.generate_images.schemas import (
    CompletedEvent,
    GeneratedImage,
    GenerateImagesRequest,
    GenerateImagesResponse,
    ProgressEvent,
)
from pydantic import ValidationError
from tests.ouranos_ml.shared.factories.images_factories import make_generate_images_request


def test_generate_images_request_when_only_prompt_should_apply_defaults():
    # Arrange
    payload = {"prompt": "a red fox"}

    # Act
    request = GenerateImagesRequest.model_validate(payload)

    # Assert
    assert request.model is None
    assert request.n == 1
    assert request.size is None
    assert request.save is True
    assert request.previews is False
    assert request.stream is False
    assert request.dimensions() is None


def test_generate_images_request_when_camel_case_should_populate_fields():
    # Arrange
    payload = {"prompt": "a red fox", "negativePrompt": "blurry", "cfgScale": 3.5}

    # Act
    request = GenerateImagesRequest.model_validate(payload)

    # Assert
    assert request.negative_prompt == "blurry"
    assert request.cfg_scale == 3.5


def test_generate_images_request_when_size_given_should_parse_dimensions():
    # Arrange
    request = make_generate_images_request(size="768x512")

    # Act
    dimensions = request.dimensions()

    # Assert
    assert dimensions == (768, 512)


def test_generate_images_request_when_size_auto_should_have_no_dimensions():
    # Arrange
    request = make_generate_images_request(size="auto")

    # Act
    dimensions = request.dimensions()

    # Assert
    assert dimensions is None


@pytest.mark.parametrize("size", ["1024", "1024x", "big", "1000x9", "4096x1024", "1020x1024", "128x128"])
def test_generate_images_request_when_size_invalid_should_raise(size):
    # Act & Assert
    with pytest.raises(ValidationError):
        make_generate_images_request(size=size)


@pytest.mark.parametrize("overrides", [{"prompt": ""}, {"n": 0}, {"n": 9}, {"steps": 0}, {"seed": -2}])
def test_generate_images_request_when_out_of_range_should_raise(overrides):
    # Act & Assert
    with pytest.raises(ValidationError):
        make_generate_images_request(**overrides)


def test_generate_images_response_when_serialized_should_use_openai_snake_case():
    # Arrange
    response = GenerateImagesResponse(
        created=1, model="m", data=[GeneratedImage(b64_json="abc", seed=1, size="1x1", output_format="png")]
    )

    # Act
    body = response.model_dump(by_alias=True)

    # Assert
    assert body["data"][0]["b64_json"] == "abc"
    assert body["data"][0]["output_format"] == "png"


def test_stream_events_when_built_should_default_their_type():
    # Arrange
    progress = ProgressEvent(sequence_number=0, image_index=0, percent=0.5, step_percent=0.0, total_percent=0.5)
    completed = CompletedEvent(
        sequence_number=1, image_index=0, b64_json="abc", seed=None, size="1x1", output_format="png"
    )

    # Act
    progress_body = progress.model_dump(by_alias=True)
    completed_body = completed.model_dump(by_alias=True)

    # Assert
    assert progress_body["type"] == "image_generation.progress"
    assert progress_body["preview_b64"] is None
    assert completed_body["type"] == "image_generation.completed"
