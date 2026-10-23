from ouranos_ml.shared.domain.core.base_schema import BaseSchema


class ImageLora(BaseSchema):
    """A LoRA on the SwarmUI server. It only works with models of the same ``compat_class``."""

    id: str
    title: str | None = None
    architecture: str | None = None
    compat_class: str | None = None
    default_weight: float | None = None
    trigger_phrase: str | None = None
    usage_hint: str | None = None


class ListImageLorasResponse(BaseSchema):
    """The LoRAs image requests can apply through ``loras``."""

    object: str = "list"
    data: list[ImageLora]
