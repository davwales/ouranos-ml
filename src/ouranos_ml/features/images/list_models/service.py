from typing import Any

import structlog

from ouranos_ml.features.images.list_models.schemas import ImageModel, ListImageModelsResponse
from ouranos_ml.features.images.presets import model_id, resolve_defaults
from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.infra.clients.swarmui_client import get_swarmui_client

logger = structlog.get_logger(__name__)

_MAX_FOLDER_DEPTH = 10


async def handle() -> ListImageModelsResponse:
    """Lists the image models SwarmUI can run."""
    logger.debug("image models list requested")

    try:
        client = get_swarmui_client()
        session_id = await client.new_session()
        response = await client.call("ListModels", session_id, path="", depth=_MAX_FOLDER_DEPTH)
    except Exception:
        logger.error("image models list upstream failed", exc_info=True)
        raise

    models = [_to_image_model(file) for file in response.get("files", []) if file.get("is_supported_model_format")]
    logger.debug("image models list complete", model_count=len(models))
    return ListImageModelsResponse(default_model=get_settings().swarmui_default_model, data=models)


def _to_image_model(file: dict[str, Any]) -> ImageModel:
    """Describe one SwarmUI model with the defaults ``/images/generations`` would apply to it."""
    defaults = resolve_defaults(file)
    return ImageModel(
        id=model_id(file["name"]),
        title=file.get("title"),
        architecture=file.get("architecture"),
        default_size=f"{defaults.width}x{defaults.height}" if defaults.width and defaults.height else None,
        default_steps=defaults.steps,
        default_cfg_scale=defaults.cfg_scale,
        usage_hint=file.get("usage_hint"),
        loaded=bool(file.get("loaded")),
    )
