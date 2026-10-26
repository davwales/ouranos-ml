import asyncio
import base64
from collections.abc import AsyncGenerator
from io import BytesIO

from PIL import Image, ImageOps

from ouranos_ml.features.images import generation
from ouranos_ml.features.images.edit_images.schemas import EditImagesRequest, decode_image
from ouranos_ml.features.images.generation import InitImage
from ouranos_ml.features.images.schemas import ImageGenerationEvent, ImagesResponse


async def handle(request: EditImagesRequest) -> ImagesResponse:
    """Edit the image and return the results all at once."""
    return await generation.handle(request, await asyncio.to_thread(_init_image, request))


async def open_stream(request: EditImagesRequest) -> AsyncGenerator[ImageGenerationEvent]:
    """Start the edit eagerly, then return an iterator over its progress events (see ``generation.open_stream``)."""
    return await generation.open_stream(request, await asyncio.to_thread(_init_image, request))


def _init_image(request: EditImagesRequest) -> InitImage:
    """The image and mask as SwarmUI takes them. Decoding the mask is CPU work, so this runs in a thread."""
    with decode_image(request.image) as image:
        width, height = image.size

    mask = None
    if request.mask is not None:
        with decode_image(request.mask) as mask_image:
            mask = _to_swarmui_mask(mask_image)

    return InitImage(image=request.image, width=width, height=height, mask=mask, creativity=request.strength)


def _to_swarmui_mask(mask: Image.Image) -> str:
    """A base64 grayscale PNG where white marks the area to change, which is how SwarmUI reads a mask.

    A mask with any transparent pixel is an OpenAI-style mask (transparent marks the area to change), so its alpha
    channel is inverted. Any other mask is already white-means-change and is only converted to grayscale.
    """
    gray = None
    if mask.has_transparency_data:
        transparency = ImageOps.invert(mask.convert("RGBA").getchannel("A"))
        if transparency.getbbox():
            gray = transparency

    buffer = BytesIO()
    (gray or mask.convert("L")).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
