from ouranos_ml.shared.domain.core.base_schema import BaseSchema


class ImageModel(BaseSchema):
    """An image model on the SwarmUI server, with the defaults a request without overrides gets."""

    id: str
    title: str | None = None
    architecture: str | None = None
    compat_class: str | None = None
    default_size: str | None = None
    default_steps: int | None = None
    default_cfg_scale: float | None = None
    usage_hint: str | None = None
    loaded: bool = False


class ListImageModelsResponse(BaseSchema):
    """The available image models and which one requests use when they name none."""

    object: str = "list"
    default_model: str
    data: list[ImageModel]
