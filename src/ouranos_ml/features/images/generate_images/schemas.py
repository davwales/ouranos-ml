"""Wire types for ``POST /images/generations``; the response and stream events are shared (``images/schemas.py``)."""

from ouranos_ml.features.images.schemas import ImageRequest


class GenerateImagesRequest(ImageRequest):
    """Generate images from a prompt."""
