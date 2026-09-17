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


def _payload(num_predictions: int) -> dict:
    """Build a valid forecast request body."""
    return {
        "points": [[make_forecast_point().model_dump(by_alias=True) for _ in range(30)]],
        "numPredictions": num_predictions,
    }


@pytest.mark.asyncio
async def test_forecast_endpoint_when_post_valid_should_return_200(async_client):
    # Arrange
    with patch("ouranos_ml.features.plutus.forecast.endpoint.forecast_points") as mock_forecast:
        mock_forecast.return_value = [[make_forecast_point(average_price=42.0)]]
        payload = _payload(1)

        # Act
        response = await async_client.post("/plutus/forecast", json=payload)

    # Assert
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_forecast_endpoint_when_post_should_pass_parsed_request_to_service(async_client):
    # Arrange
    with patch("ouranos_ml.features.plutus.forecast.endpoint.forecast_points") as mock_forecast:
        mock_forecast.return_value = [[make_forecast_point(average_price=10.0)]]
        payload = _payload(1)

        # Act
        await async_client.post("/plutus/forecast", json=payload)

    # Assert
    args = mock_forecast.call_args.args
    assert len(args[0]) == 1
    assert len(args[0][0]) == 30
    assert args[0][0][0].average_price == make_forecast_point().average_price
    assert args[1] == 1


@pytest.mark.asyncio
async def test_forecast_endpoint_when_post_should_return_snake_case_wire_format(async_client):
    # Arrange
    with patch("ouranos_ml.features.plutus.forecast.endpoint.forecast_points") as mock_forecast:
        mock_forecast.return_value = [[make_forecast_point(average_price=10.0)]]
        payload = _payload(1)

        # Act
        response = await async_client.post("/plutus/forecast", json=payload)

    # Assert
    body = response.json()
    assert "average_price" in body[0][0]
    assert "averagePrice" not in body[0][0]


@pytest.mark.asyncio
async def test_forecast_endpoint_when_num_predictions_zero_should_return_422(async_client):
    # Arrange
    payload = _payload(0)

    # Act
    response = await async_client.post("/plutus/forecast", json=payload)

    # Assert
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_forecast_endpoint_when_num_predictions_negative_should_return_422(async_client):
    # Arrange
    payload = _payload(-1)

    # Act
    response = await async_client.post("/plutus/forecast", json=payload)

    # Assert
    assert response.status_code == 422


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