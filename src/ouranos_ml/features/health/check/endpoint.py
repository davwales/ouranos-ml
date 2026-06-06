from fastapi import APIRouter

from ouranos_ml.features.health.check.schemas import HealthResponse
from ouranos_ml.features.health.check.service import handle


async def _health_check() -> HealthResponse:
    """Returns the health status of the service."""
    return await handle()


def register(router: APIRouter) -> None:
    """Register health check endpoints on the provided router."""
    router.get("")(_health_check)