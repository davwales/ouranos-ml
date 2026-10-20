from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI

from ouranos_ml.features.health.check.schemas import HealthCheck, HealthResponse, HealthStatus
from ouranos_ml.features.health.router import health_router
from tests.ouranos_ml.shared.factories.health_factories import make_health_check, make_health_response


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
        status=HealthStatus.HEALTHY,
        checks={"llm": HealthCheck(status=HealthStatus.HEALTHY, description="ok")},
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
        status=HealthStatus.HEALTHY,
        checks={"llm": HealthCheck(status=HealthStatus.HEALTHY, description="ok")},
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


@pytest.mark.asyncio
async def test_health_endpoint_when_unhealthy_should_return_503_with_health_body(async_client):
    # Arrange
    mock_response = make_health_response(
        status=HealthStatus.UNHEALTHY,
        checks={"llm": make_health_check(status=HealthStatus.UNHEALTHY, description="LLM backend is unreachable")},
    )
    with patch(
        "ouranos_ml.features.health.check.endpoint.handle",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        # Act
        response = await async_client.get("/health")

    # Assert
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["checks"]["llm"]["status"] == "unhealthy"


@pytest.mark.asyncio
@pytest.mark.parametrize("service_status", [HealthStatus.DEGRADED, HealthStatus.NOT_CONFIGURED])
async def test_health_endpoint_when_degraded_or_not_configured_should_return_200(async_client, service_status):
    # Arrange
    mock_response = make_health_response(status=service_status)
    with patch(
        "ouranos_ml.features.health.check.endpoint.handle",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        # Act
        response = await async_client.get("/health")

    # Assert
    assert response.status_code == 200
    assert response.json()["status"] == service_status.value
