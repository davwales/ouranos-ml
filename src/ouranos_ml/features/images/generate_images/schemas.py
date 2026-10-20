"""Wire types for ``POST /images/generations``: an OpenAI Images-style request/response plus our progress events."""

import re
from typing import Annotated, Any, Literal

from ouranos_ml.shared.domain.core.base_schema import BaseSchema
from pydantic import Field, field_validator

_SIZE_PATTERN = re.compile(r"^(\d+)x(\d+)$")
MIN_DIMENSION = 256
MAX_DIMENSION = 2048


class GenerateImagesRequest(BaseSchema):
    """Generate images from a prompt. Only ``prompt`` is required; everything else has a model-aware default."""

    prompt: str = Field(min_length=1)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=8)
    size: str | None = None
    negative_prompt: str | None = None
    steps: int | None = Field(default=None, ge=1, le=200)
    cfg_scale: float | None = Field(default=None, ge=0, le=100)
    seed: int | None = Field(default=None, ge=-1)
    sampler: str | None = None
    scheduler: str | None = None
    save: bool = True
    previews: bool = False
    stream: bool = False
    params: dict[str, Any] | None = None

    @field_validator("size")
    @classmethod
    def _validate_size(cls, size: str | None) -> str | None:
        """Accept ``auto`` or ``WIDTHxHEIGHT`` with both sides in range and multiples of 8."""
        if size is None or size == "auto":
            return size

        match = _SIZE_PATTERN.match(size)
        if not match:
            raise ValueError("size must be 'auto' or 'WIDTHxHEIGHT', e.g. '1024x1024'")

        for dimension in map(int, match.groups()):
            if not MIN_DIMENSION <= dimension <= MAX_DIMENSION or dimension % 8:
                raise ValueError(f"size dimensions must be multiples of 8 between {MIN_DIMENSION} and {MAX_DIMENSION}")

        return size

    def dimensions(self) -> tuple[int, int] | None:
        """The explicit ``(width, height)``, or ``None`` to use the model's standard size."""
        if self.size is None or self.size == "auto":
            return None

        width, height = self.size.split("x")
        return int(width), int(height)


class GeneratedImage(BaseSchema):
    """A finished image, inline as base64."""

    b64_json: str
    seed: int | None = None
    size: str | None = None
    output_format: str


class GenerateImagesResponse(BaseSchema):
    """The non-streaming result: OpenAI's ``ImagesResponse`` core (``created``, ``data[].b64_json``) plus extras."""

    created: int
    model: str
    data: list[GeneratedImage]


class StatusEvent(BaseSchema):
    """The generation moved to a new stage; sent only when the stage changes."""

    type: Literal["image_generation.status"] = "image_generation.status"
    sequence_number: int
    stage: Literal["queued", "loading_model", "generating"]
    waiting_gens: int
    loading_models: int
    live_gens: int


class ProgressEvent(BaseSchema):
    """Progress on one image; ``total_percent`` (0-1, across all images) is the number to show users."""

    type: Literal["image_generation.progress"] = "image_generation.progress"
    sequence_number: int
    image_index: int
    percent: float
    step_percent: float
    total_percent: float
    preview_b64: str | None = None


class CompletedEvent(GeneratedImage):
    """One image finished."""

    type: Literal["image_generation.completed"] = "image_generation.completed"
    sequence_number: int
    image_index: int


class DoneEvent(BaseSchema):
    """Every image finished; the stream ends after this event."""

    type: Literal["image_generation.done"] = "image_generation.done"
    sequence_number: int
    created: int
    model: str
    count: int


class ErrorEvent(BaseSchema):
    """An error that ended the stream."""

    type: Literal["error"] = "error"
    sequence_number: int
    code: str | None = None
    message: str


ImageGenerationEvent = Annotated[
    StatusEvent | ProgressEvent | CompletedEvent | DoneEvent | ErrorEvent,
    Field(discriminator="type"),
]
