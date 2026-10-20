from fastapi import APIRouter, Response, status

from ouranos_ml.features.health.check.schemas import HealthResponse, ServiceStatus
from ouranos_ml.features.health.check.service import handle


async def _health_check(response: Response) -> HealthResponse:
    """Returns the health status of the service.

    Only an unhealthy aggregate returns 503, so orchestrators can act on it; degraded
    (e.g. CPU-only, SwarmUI down) and not-configured states stay 200.
    """
    result = await handle()
    if result.status == ServiceStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


def register(router: APIRouter) -> None:
    """Register health check endpoints on the provided router."""
    router.get(
        "",
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse, "description": "Service is unhealthy"}
        },
    )(_health_check)
