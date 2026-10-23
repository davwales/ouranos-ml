from fastapi import APIRouter

from ouranos_ml.features.images.list_loras.schemas import ListImageLorasResponse
from ouranos_ml.features.images.list_loras.service import handle


async def _list_loras() -> ListImageLorasResponse:
    """Lists the LoRAs available to image requests."""
    return await handle()


def register(router: APIRouter) -> None:
    """Register list-image-loras endpoints on the provided router."""
    router.get("/loras")(_list_loras)
