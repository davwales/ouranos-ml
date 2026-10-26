from fastapi import APIRouter

from ouranos_ml.features.images.edit_images.endpoint import register as register_edit_images
from ouranos_ml.features.images.generate_images.endpoint import register as register_generate_images
from ouranos_ml.features.images.list_loras.endpoint import register as register_list_loras
from ouranos_ml.features.images.list_models.endpoint import register as register_list_models

images_router = APIRouter(prefix="/images")
register_generate_images(images_router)
register_edit_images(images_router)
register_list_models(images_router)
register_list_loras(images_router)
