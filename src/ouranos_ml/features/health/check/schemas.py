from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ouranos_ml.shared.domain.core.base_schema import BaseSchema
from pydantic import Field


class CheckStatus(StrEnum):
    """Status of an individual health check."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    NOT_CONFIGURED = "not_configured"


class ServiceStatus(StrEnum):
    """Overall health status of the service."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    NOT_CONFIGURED = "not_configured"


class GpuCheckData(BaseSchema):
    """Structured data payload for the GPU health check."""

    device_count: int


class HealthCheck(BaseSchema):
    """Schema for a single health check result."""

    status: CheckStatus
    description: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    data: dict[str, Any] | None = None


class HealthResponse(BaseSchema):
    """Schema for the aggregate health check response."""

    status: ServiceStatus
    checks: dict[str, HealthCheck]

