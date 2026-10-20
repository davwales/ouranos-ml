from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ouranos_ml.shared.domain.core.base_schema import BaseSchema
from pydantic import Field


class HealthStatus(StrEnum):
    """Status of an individual health check, and of the service as a whole."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    NOT_CONFIGURED = "not_configured"


class GpuCheckData(BaseSchema):
    """Structured data payload for the GPU health check."""

    device_count: int


class SwarmUICheckData(BaseSchema):
    """Structured data payload for the SwarmUI health check."""

    backend_status: str


class PlutusCheckData(BaseSchema):
    """Structured data payload for the Plutus forecast model health check."""

    model_name: str
    sequence_length: int


class HealthCheck(BaseSchema):
    """Schema for a single health check result."""

    status: HealthStatus
    description: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    data: dict[str, Any] | None = None


class HealthResponse(BaseSchema):
    """Schema for the aggregate health check response."""

    status: HealthStatus
    checks: dict[str, HealthCheck]
