"""Wire types for ``POST /images/edits``: an image request plus the image to start from and an optional mask."""

import base64
import binascii
from io import BytesIO
from typing import Self

from PIL import Image, UnidentifiedImageError

from ouranos_ml.features.images.schemas import ImageRequest
from pydantic import Field, field_validator, model_validator

_FORMATS = ("PNG", "JPEG", "WEBP")


def decode_image(data: str) -> Image.Image:
    """Open a base64 PNG, JPEG or WebP image; only the header is read until the pixels are used."""
    return Image.open(BytesIO(base64.b64decode(data, validate=True)), formats=_FORMATS)


class EditImagesRequest(ImageRequest):
    """Edit an image: rework all of it from the prompt (img2img), or only the area a mask marks (inpainting).

    ``image`` and ``mask`` are base64 or data URLs of a PNG, JPEG or WebP image. In the mask, white marks the area to
    change and black the area to keep. A mask with transparent pixels follows OpenAI's convention instead: transparent
    marks the area to change. Without ``size``, the output keeps the image's aspect ratio at the model's standard
    resolution. ``strength`` is how far to move away from the image (SwarmUI's init image creativity, default 0.6).
    """

    image: str = Field(min_length=1)
    mask: str | None = None
    strength: float | None = Field(default=None, ge=0, le=1)

    @field_validator("image", "mask")
    @classmethod
    def _validate_image(cls, value: str | None) -> str | None:
        """Accept base64 or a data URL of a readable image, normalized to bare base64."""
        if value is None:
            return None

        data = value.split(",", 1)[-1] if value.startswith("data:") else value
        try:
            with decode_image(data):
                pass
        except (binascii.Error, ValueError, UnidentifiedImageError):
            raise ValueError("must be a base64-encoded PNG, JPEG or WebP image")

        return data

    @model_validator(mode="after")
    def _validate_mask_size(self) -> Self:
        """The mask has to line up with the image pixel for pixel."""
        if self.mask is not None:
            with decode_image(self.image) as image, decode_image(self.mask) as mask:
                if image.size != mask.size:
                    raise ValueError(f"mask is {mask.width}x{mask.height} but image is {image.width}x{image.height}")
        return self
