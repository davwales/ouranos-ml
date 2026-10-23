import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ouranos_ml.features.images.edit_images.service import handle, open_stream
from tests.ouranos_ml.shared.factories.images_factories import (
    PNG_B64,
    make_edit_images_request,
    make_image_b64,
    make_swarmui_client,
)

GET_CLIENT = "ouranos_ml.features.images.generation.get_swarmui_client"
GET_SETTINGS = "ouranos_ml.features.images.generation.get_settings"


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.swarmui_default_model = "flux-2-klein-9b"
    settings.swarmui_generation_timeout_seconds = 600.0
    return settings


def _sent_payload(client: MagicMock) -> dict:
    return client.open_generation.await_args.args[1]


def _sent_mask(client: MagicMock) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(_sent_payload(client)["maskimage"])))


@pytest.mark.asyncio
async def test_handle_when_edit_succeeds_should_send_init_image_and_return_images():
    # Arrange
    client = make_swarmui_client()
    request = make_edit_images_request(strength=0.8)

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        response = await handle(request)

    # Assert
    payload = _sent_payload(client)
    assert payload["initimage"] == request.image
    assert payload["initimagecreativity"] == 0.8
    assert (payload["width"], payload["height"]) == (1184, 880)
    assert "maskimage" not in payload
    assert response.data[0].b64_json == PNG_B64


@pytest.mark.asyncio
async def test_open_stream_when_edit_succeeds_should_stream_generation_events():
    # Arrange
    client = make_swarmui_client()
    request = make_edit_images_request(stream=True)

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        events = [event async for event in await open_stream(request)]

    # Assert
    assert _sent_payload(client)["initimage"] == request.image
    assert events[-1].type == "image_generation.done"


@pytest.mark.asyncio
async def test_handle_when_mask_is_black_and_white_should_send_it_as_grayscale_png():
    # Arrange
    client = make_swarmui_client()
    request = make_edit_images_request(mask=make_image_b64(64, 48, mode="RGB", color="white"))

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(request)

    # Assert
    mask = _sent_mask(client)
    assert (mask.format, mask.mode, mask.size) == ("PNG", "L", (64, 48))
    assert mask.getextrema() == (255, 255)


@pytest.mark.asyncio
async def test_handle_when_mask_has_transparency_should_mark_transparent_pixels_for_change():
    # Arrange
    rgba = Image.new("RGBA", (64, 48), (0, 0, 0, 255))
    rgba.paste((0, 0, 0, 0), (0, 0, 32, 48))
    buffer = BytesIO()
    rgba.save(buffer, format="PNG")
    client = make_swarmui_client()
    request = make_edit_images_request(mask=base64.b64encode(buffer.getvalue()).decode("ascii"))

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(request)

    # Assert
    mask = _sent_mask(client)
    assert mask.getpixel((0, 0)) == 255
    assert mask.getpixel((63, 0)) == 0


@pytest.mark.asyncio
async def test_handle_when_mask_has_alpha_but_is_opaque_should_use_its_brightness():
    # Arrange
    client = make_swarmui_client()
    request = make_edit_images_request(mask=make_image_b64(64, 48, mode="RGBA", color=(255, 255, 255, 255)))

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(request)

    # Assert
    assert _sent_mask(client).getextrema() == (255, 255)
