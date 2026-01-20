from fastapi import APIRouter

models_router = APIRouter(prefix="/models")

from ouranos_ml.features.models.list_models.api import list_models
