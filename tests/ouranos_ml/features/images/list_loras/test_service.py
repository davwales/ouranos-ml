from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ouranos_ml.features.images.list_loras.service import handle
from ouranos_ml.shared.infra.swarmui.errors import SwarmUIUnavailableError
from tests.ouranos_ml.shared.factories.images_factories import make_lora_description

GET_CLIENT = "ouranos_ml.features.images.list_loras.service.get_swarmui_client"


def _client(files: list[dict]) -> MagicMock:
    client = MagicMock()
    client.call_shared = AsyncMock(return_value={"folders": [], "files": files})
    return client


@pytest.mark.asyncio
async def test_handle_when_loras_listed_should_describe_them():
    # Arrange
    files = [
        make_lora_description(default_weight="0.8", trigger_phrase="ink wash style"),
        make_lora_description(name="plain.safetensors", compat_class="", architecture=None),
    ]
    client = _client(files)

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await handle()

    # Assert
    client.call_shared.assert_awaited_once_with("ListModels", path="", depth=10, subtype="LoRA")
    ink, plain = response.data
    assert ink.id == "styles/ink-wash"
    assert (ink.compat_class, ink.default_weight, ink.trigger_phrase) == ("flux-2-klein-9b", 0.8, "ink wash style")
    assert plain.id == "plain"
    assert (plain.compat_class, plain.architecture, plain.default_weight, plain.trigger_phrase) == (
        None,
        None,
        None,
        None,
    )


@pytest.mark.asyncio
async def test_handle_when_lora_format_unsupported_should_leave_it_out():
    # Arrange
    client = _client([make_lora_description(is_supported_model_format=False)])

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await handle()

    # Assert
    assert response.data == []


@pytest.mark.asyncio
async def test_handle_when_swarmui_unreachable_should_reraise():
    # Arrange
    client = _client([])
    client.call_shared.side_effect = SwarmUIUnavailableError("down")

    # Act & Assert
    with patch(GET_CLIENT, return_value=client), pytest.raises(SwarmUIUnavailableError):
        await handle()
