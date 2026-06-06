from fastapi import APIRouter

from ouranos_ml.features.models.list_models.endpoint import register as register_list_models

models_router = APIRouter(prefix="/models")
register_list_models(models_router)