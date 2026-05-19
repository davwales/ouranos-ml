from ouranos_ml.features.health.check.handler import handle
from ouranos_ml.features.health.check.schemas import HealthResponse
from ouranos_ml.features.health.router import health_router


@health_router.get("")
async def health_check() -> HealthResponse:
    """Returns the health status of the service."""
    return await handle()
