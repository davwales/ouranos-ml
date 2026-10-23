"""Factory functions for image generation test data.

SwarmUI messages are built in the exact wire shape ``GenerateText2ImageWS`` sends (captured from a live server),
so the service's translation is exercised as it is against a real SwarmUI.
"""

import base64
import json
from collections.abc import AsyncGenerator, Iterable
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from ouranos_ml.features.images.edit_images.schemas import EditImagesRequest
from ouranos_ml.features.images.generate_images.schemas import GenerateImagesRequest
from ouranos_ml.shared.infra.swarmui.errors import SwarmUIError

PNG_B64 = "iVBORw0KGgo="


def make_generate_images_request(**overrides: Any) -> GenerateImagesRequest:
    """Create a GenerateImagesRequest from wire-format fields with sensible defaults."""
    payload: dict[str, Any] = {"prompt": "a red fox in snow"}
    payload.update(overrides)
    return GenerateImagesRequest.model_validate(payload)


def make_image_b64(
    width: int = 64,
    height: int = 48,
    *,
    mode: str = "RGB",
    color: str | int | tuple[int, ...] = "red",
    image_format: str = "PNG",
) -> str:
    """Create a solid-color image as base64, the way edit requests carry images and masks."""
    buffer = BytesIO()
    Image.new(mode, (width, height), color).save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_edit_images_request(**overrides: Any) -> EditImagesRequest:
    """Create an EditImagesRequest from wire-format fields, editing a 64x48 image by default."""
    payload: dict[str, Any] = {"prompt": "make it winter", "image": make_image_b64()}
    payload.update(overrides)
    return EditImagesRequest.model_validate(payload)


def make_model_description(
    *,
    name: str = "flux-2-klein-9b.safetensors",
    architecture: str = "flux.2-klein-9b",
    compat_class: str = "flux-2-klein-9b",
    width: int = 1024,
    height: int = 1024,
    **overrides: Any,
) -> dict[str, Any]:
    """Create a SwarmUI model description, as returned by ``DescribeModel`` and listed by ``ListModels``."""
    description: dict[str, Any] = {
        "name": name,
        "title": name.rsplit(".", 1)[0],
        "architecture": architecture,
        "compat_class": compat_class,
        "standard_width": width,
        "standard_height": height,
        "usage_hint": None,
        "loaded": False,
        "is_supported_model_format": True,
    }
    description.update(overrides)
    return description


def make_lora_description(
    *,
    name: str = "styles/ink-wash.safetensors",
    compat_class: str = "flux-2-klein-9b",
    default_weight: str = "",
    **overrides: Any,
) -> dict[str, Any]:
    """Create a SwarmUI LoRA description, as ``DescribeModel``/``ListModels`` return with ``subtype=LoRA``.

    SwarmUI sends unset metadata as empty strings, including ``lora_default_weight``.
    """
    description: dict[str, Any] = {
        "name": name,
        "title": name.rsplit(".", 1)[0],
        "architecture": f"{compat_class}/lora",
        "compat_class": compat_class,
        "lora_default_weight": default_weight,
        "trigger_phrase": "",
        "usage_hint": None,
        "is_supported_model_format": True,
    }
    description.update(overrides)
    return description


def make_swarm_status(*, waiting_gens: int = 0, loading_models: int = 0, live_gens: int = 0) -> dict[str, Any]:
    """Create a SwarmUI ``status`` message."""
    return {
        "status": {
            "waiting_gens": waiting_gens,
            "loading_models": loading_models,
            "waiting_backends": 0,
            "live_gens": live_gens,
        },
        "backend_status": {"status": "running", "class": "", "message": "", "any_loading": False},
        "supported_features": ["comfyui"],
    }


def make_swarm_progress(
    *, batch_index: int = 0, overall_percent: float = 0.5, current_percent: float = 0.0, preview: str | None = None
) -> dict[str, Any]:
    """Create a SwarmUI ``gen_progress`` message."""
    progress: dict[str, Any] = {
        "batch_index": str(batch_index),
        "request_id": "1",
        "overall_percent": overall_percent,
        "current_percent": current_percent,
    }
    if preview is not None:
        progress["preview"] = preview
    return {"gen_progress": progress}


def make_swarm_image(
    *, batch_index: int = 0, image: str = f"data:image/png;base64,{PNG_B64}", seed: int = 42, width: int = 1024
) -> dict[str, Any]:
    """Create a SwarmUI ``image`` message; ``image`` is a data URL, or a ``View/`` path when saved."""
    metadata = {"sui_image_params": {"prompt": "a red fox in snow", "width": width, "height": width, "seed": seed}}
    return {
        "image": image,
        "batch_index": str(batch_index),
        "request_id": "1",
        "metadata": json.dumps(metadata, indent=2),
    }


def make_swarm_error(message: str = "Invalid value for parameter Sampler") -> dict[str, Any]:
    """Create a SwarmUI ``error`` message."""
    return {"error": message, "request_id": "1"}


def make_swarm_messages(*, n: int = 1) -> list[dict[str, Any]]:
    """Create a happy-path SwarmUI message sequence for ``n`` images, with the noise a real server sends."""
    messages = [
        make_swarm_status(waiting_gens=n),
        make_swarm_status(waiting_gens=n, loading_models=1),
        make_swarm_status(waiting_gens=n, loading_models=1),
        make_swarm_status(waiting_gens=n, live_gens=1),
    ]
    for index in range(n):
        messages += [
            make_swarm_progress(batch_index=index, overall_percent=0.5),
            make_swarm_progress(batch_index=index, overall_percent=1.0, current_percent=1.0),
            make_swarm_image(batch_index=index, seed=42 + index),
        ]
    messages += [{"keep_alive": True}, make_swarm_status(), {"discard_indices": []}]
    return messages


async def make_message_stream(messages: Iterable[dict[str, Any]]) -> AsyncGenerator[dict[str, Any]]:
    """Yield SwarmUI messages as the async generator ``SwarmUIClient.open_generation`` returns."""
    for message in messages:
        yield message


def make_swarmui_client(
    *,
    messages: list[dict[str, Any]] | None = None,
    description: dict[str, Any] | None = None,
    loras: Iterable[dict[str, Any]] = (),
) -> MagicMock:
    """Create a mock ``SwarmUIClient`` that describes ``description`` and ``loras`` and streams ``messages``.

    LoRAs are found by their name with or without the extension, as SwarmUI does; others are "Model not found.".
    """
    model = description or make_model_description()
    loras_by_name = {}
    for lora in loras:
        loras_by_name[lora["name"]] = loras_by_name[lora["name"].rsplit(".", 1)[0]] = lora

    async def describe(route: str, session_id: str, *, modelName: str, subtype: str) -> dict[str, Any]:  # noqa: N803
        if subtype != "LoRA":
            return {"model": model}
        if modelName not in loras_by_name:
            raise SwarmUIError("Model not found.")
        return {"model": loras_by_name[modelName]}

    client = MagicMock()
    client.new_session = AsyncMock(return_value="session-1")
    client.call = AsyncMock(side_effect=describe)
    stream = make_message_stream(make_swarm_messages() if messages is None else messages)
    client.open_generation = AsyncMock(return_value=stream)
    client.interrupt = AsyncMock()
    client.fetch_file = AsyncMock(return_value=b"\x89PNG")
    return client
