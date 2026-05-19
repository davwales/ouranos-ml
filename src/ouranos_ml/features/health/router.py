from fastapi import APIRouter

health_router = APIRouter(prefix="/health")

# Routes
from ouranos_ml.features.health.check.api import health_check
