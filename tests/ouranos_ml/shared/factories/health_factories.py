"""Factory functions for creating health check test data."""

from datetime import UTC, datetime

from ouranos_ml.features.health.check.schemas import (
    CheckStatus,
    HealthCheck,
    HealthResponse,
    ServiceStatus,
)


def make_health_check(
    *,
    status: CheckStatus = CheckStatus.HEALTHY,
    description: str = "Service is healthy",
    timestamp: datetime | None = None,
    data: dict | None = None,
) -> HealthCheck:
    """Create a HealthCheck with sensible defaults."""
    return HealthCheck(
        status=status,
        description=description,
        timestamp=timestamp if timestamp is not None else datetime.now(tz=UTC),
        data=data,
    )


def make_health_response(
    *,
    status: ServiceStatus = ServiceStatus.HEALTHY,
    checks: dict[str, HealthCheck] | None = None,
) -> HealthResponse:
    """Create a HealthResponse with sensible defaults."""
    return HealthResponse(
        status=status,
        checks=checks
        or {
            "llm": make_health_check(status=CheckStatus.HEALTHY, description="LLM backend is reachable"),
            "gpu": make_health_check(
                status=CheckStatus.HEALTHY,
                description="GPU is available",
                data={"device_count": 1},
            ),
        },
    )
