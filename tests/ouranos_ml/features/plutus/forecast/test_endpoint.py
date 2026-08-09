from unittest.mock import patch

import pytest
from fastapi import FastAPI

from ouranos_ml.features.plutus.router import plutus_router
from tests.ouranos_ml.shared.factories.forecast_factories import make_forecast_point


@pytest.fixture
def app() -> FastAPI:
    """FastAPI app mounting only the plutus router (mocked torch via root conftest)."""
    application = FastAPI()
    application.include_router(plutus_router)
    return application


@pytest.mark.asyncio
async def test_forecast_endpoint_when_post_valid_should_return_200(async_client):
    # Arrange
    with patch("ouranos_ml.features.plutus.forecast.endpoint.forecast_points") as mock_forecast:
        mock_forecast.return_value = [[make_forecast_point(average_price=42.0)]]
        payload = {
            "points": [[make_forecast_point().model_dump(by_alias=True) for _ in range(30)]],
            "numPredictions": 1,
        }

        # Act
        response = await async_client.post("/plutus/forecast", json=payload)

    # Assert
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_forecast_endpoint_when_post_should_return_forecast_shape(async_client):
    # Arrange
    with patch("ouranos_ml.features.plutus.forecast.endpoint.forecast_points") as mock_forecast:
        mock_forecast.return_value = [
            [make_forecast_point(average_price=10.0), make_forecast_point(average_price=20.0)]
        ]
        payload = {
            "points": [[make_forecast_point().model_dump(by_alias=True) for _ in range(30)]],
            "numPredictions": 2,
        }

        # Act
        response = await async_client.post("/plutus/forecast", json=payload)

    # Assert
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert isinstance(body[0], list)
    assert len(body[0]) == 2


@pytest.mark.asyncio
async def test_forecast_endpoint_when_missing_points_should_return_422(async_client):
    # Arrange
    payload = {"numPredictions": 3}

    # Act
    response = await async_client.post("/plutus/forecast", json=payload)

    # Assert
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_forecast_endpoint_when_missing_num_predictions_should_return_422(async_client):
    # Arrange
    sequence = [make_forecast_point().model_dump(by_alias=True) for _ in range(30)]
    payload = {"points": [sequence]}

    # Act
    response = await async_client.post("/plutus/forecast", json=payload)

    # Assert
    assert response.status_code == 422
