from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI

from ouranos_ml.features.health.check.schemas import CheckStatus, HealthCheck, HealthResponse, ServiceStatus
from ouranos_ml.features.health.router import health_router


@pytest.fixture
def app() -> FastAPI:
    """FastAPI app mounting only the health router (isolation from torch-dependent routers)."""
    application = FastAPI()
    application.include_router(health_router)
    return application


@pytest.mark.asyncio
async def test_health_endpoint_when_get_should_return_200(async_client):
    # Arrange
    mock_response = HealthResponse(
        status=ServiceStatus.HEALTHY,
        checks={"llm": HealthCheck(status=CheckStatus.HEALTHY, description="ok")},
    )
    with patch(
        "ouranos_ml.features.health.check.endpoint.handle",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        # Act
        response = await async_client.get("/health")

    # Assert
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoint_when_get_should_return_health_response_shape(async_client):
    # Arrange
    mock_response = HealthResponse(
        status=ServiceStatus.HEALTHY,
        checks={"llm": HealthCheck(status=CheckStatus.HEALTHY, description="ok")},
    )
    with patch(
        "ouranos_ml.features.health.check.endpoint.handle",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        # Act
        response = await async_client.get("/health")

    # Assert
    body = response.json()
    assert "status" in body
    assert "checks" in body
    assert body["status"] == "healthy"
    assert "llm" in body["checks"]
