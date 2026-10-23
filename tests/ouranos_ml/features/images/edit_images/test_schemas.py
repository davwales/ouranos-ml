import pytest

from pydantic import ValidationError
from tests.ouranos_ml.shared.factories.images_factories import make_edit_images_request, make_image_b64


def test_edit_images_request_when_data_url_given_should_store_bare_base64():
    # Arrange
    image = make_image_b64()

    # Act
    request = make_edit_images_request(image=f"data:image/png;base64,{image}")

    # Assert
    assert request.image == image


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP"])
def test_edit_images_request_when_supported_format_should_accept_it(image_format):
    # Arrange
    image = make_image_b64(image_format=image_format)

    # Act
    request = make_edit_images_request(image=image)

    # Assert
    assert request.image == image


@pytest.mark.parametrize("image", ["", "not base64!", "aGVsbG8gd29ybGQ=", make_image_b64(image_format="GIF")])
def test_edit_images_request_when_image_unreadable_should_raise(image):
    # Act & Assert
    with pytest.raises(ValidationError, match="image"):
        make_edit_images_request(image=image)


def test_edit_images_request_when_mask_matches_image_should_accept_it():
    # Arrange
    mask = make_image_b64(64, 48, mode="L", color=255)

    # Act
    request = make_edit_images_request(mask=mask)

    # Assert
    assert request.mask == mask


def test_edit_images_request_when_mask_size_differs_should_raise():
    # Act & Assert
    with pytest.raises(ValidationError, match="mask is 32x32 but image is 64x48"):
        make_edit_images_request(mask=make_image_b64(32, 32))


@pytest.mark.parametrize("strength", [-0.1, 1.1])
def test_edit_images_request_when_strength_out_of_range_should_raise(strength):
    # Act & Assert
    with pytest.raises(ValidationError):
        make_edit_images_request(strength=strength)


def test_edit_images_request_when_shared_fields_given_should_accept_them():
    # Act
    request = make_edit_images_request(size="512x512", loras=[{"name": "detail"}], stream=True)

    # Assert
    assert request.dimensions() == (512, 512)
    assert request.loras[0].name == "detail"
    assert request.stream is True
