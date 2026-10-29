from typing import Any

import structlog

from ouranos_ml.features.images.list_loras.schemas import ImageLora, ListImageLorasResponse
from ouranos_ml.features.images.presets import model_id
from ouranos_ml.shared.infra.clients.swarmui_client import get_swarmui_client

logger = structlog.get_logger(__name__)

_MAX_FOLDER_DEPTH = 10


async def handle() -> ListImageLorasResponse:
    """Lists the LoRAs SwarmUI can apply."""
    logger.debug("image loras list requested")

    try:
        client = get_swarmui_client()
        response = await client.call_shared("ListModels", path="", depth=_MAX_FOLDER_DEPTH, subtype="LoRA")
    except Exception:
        logger.error("image loras list upstream failed", exc_info=True)
        raise

    loras = [_to_image_lora(file) for file in response.get("files", []) if file.get("is_supported_model_format")]
    logger.debug("image loras list complete", lora_count=len(loras))
    return ListImageLorasResponse(data=loras)


def _to_image_lora(file: dict[str, Any]) -> ImageLora:
    """Describe one SwarmUI LoRA. SwarmUI sends empty strings for unset metadata."""
    try:
        default_weight = float(file["lora_default_weight"])
    except (KeyError, TypeError, ValueError):
        default_weight = None

    return ImageLora(
        id=model_id(file["name"]),
        title=file.get("title") or None,
        architecture=file.get("architecture") or None,
        compat_class=file.get("compat_class") or None,
        default_weight=default_weight,
        trigger_phrase=file.get("trigger_phrase") or None,
        usage_hint=file.get("usage_hint") or None,
    )
