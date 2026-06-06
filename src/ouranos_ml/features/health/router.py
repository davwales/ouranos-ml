from fastapi import APIRouter

from ouranos_ml.features.health.check.endpoint import register as register_check

health_router = APIRouter(prefix="/health")
register_check(health_router)