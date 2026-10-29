from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ouranos_ml.features.images.list_models.service import handle
from ouranos_ml.shared.infra.swarmui.errors import SwarmUIUnavailableError
from tests.ouranos_ml.shared.factories.images_factories import make_model_description

GET_CLIENT = "ouranos_ml.features.images.list_models.service.get_swarmui_client"


def _client(files: list[dict]) -> MagicMock:
    client = MagicMock()
    client.call_shared = AsyncMock(return_value={"folders": [], "files": files})
    return client


@pytest.mark.asyncio
async def test_handle_when_models_listed_should_describe_them_with_their_defaults():
    # Arrange
    files = [
        make_model_description(
            loaded=True,
            usage_hint="few steps",
        ),
        make_model_description(
            name="Other/mystery.safetensors",
            architecture="mystery",
            compat_class="mystery",
            width=512,
            height=768,
        ),
    ]

    client = _client(files)

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await handle()

    # Assert
    client.call_shared.assert_awaited_once_with("ListModels", path="", depth=10)
    klein, mystery = response.data
    assert klein.id == "flux-2-klein-9b"
    assert klein.compat_class == "flux-2-klein-9b"
    assert (klein.default_size, klein.default_steps, klein.default_cfg_scale) == ("1024x1024", 10, 1.0)
    assert klein.loaded is True
    assert klein.usage_hint == "few steps"
    assert mystery.id == "Other/mystery"
    assert (mystery.default_size, mystery.default_steps, mystery.default_cfg_scale) == ("512x768", None, None)
    assert response.default_model == "flux-2-klein-9b"


@pytest.mark.asyncio
async def test_handle_when_model_format_unsupported_should_leave_it_out():
    # Arrange
    client = _client([make_model_description(is_supported_model_format=False)])

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await handle()

    # Assert
    assert response.data == []


@pytest.mark.asyncio
async def test_handle_when_model_has_no_standard_size_should_report_no_default_size():
    # Arrange
    client = _client([make_model_description(width=0, height=0)])

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await handle()

    # Assert
    assert response.data[0].default_size is None


@pytest.mark.asyncio
async def test_handle_when_swarmui_unreachable_should_reraise():
    # Arrange
    client = _client([])
    client.call_shared.side_effect = SwarmUIUnavailableError("down")

    # Act & Assert
    with patch(GET_CLIENT, return_value=client), pytest.raises(SwarmUIUnavailableError):
        await handle()
