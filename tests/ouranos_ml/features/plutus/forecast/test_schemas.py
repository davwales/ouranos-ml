import pytest

from ouranos_ml.features.plutus.forecast.schemas import ForecastRequest
from pydantic import ValidationError
from tests.ouranos_ml.shared.factories.forecast_factories import (
    make_forecast_point,
    make_forecast_request,
    make_sequence,
)


def test_forecast_request_when_valid_should_accept():
    # Arrange
    points = [make_sequence(30)]

    # Act
    request = ForecastRequest(points=points, num_predictions=3)

    # Assert
    assert request.points == points
    assert request.num_predictions == 3


def test_forecast_request_when_points_missing_should_raise_validation_error():
    # Arrange

    # Act & Assert
    with pytest.raises(ValidationError):
        ForecastRequest(num_predictions=3)


def test_forecast_request_when_num_predictions_missing_should_raise_validation_error():
    # Arrange
    points = [make_sequence(30)]

    # Act & Assert
    with pytest.raises(ValidationError):
        ForecastRequest(points=points)


def test_forecast_request_when_serialized_should_use_camel_case():
    # Arrange
    request = make_forecast_request(num_predictions=5)

    # Act
    data = request.model_dump(by_alias=True)

    # Assert
    assert "numPredictions" in data
    assert "num_predictions" not in data


def test_plutus_forecast_point_when_serialized_should_use_camel_case():
    # Arrange
    point = make_forecast_point()

    # Act
    data = point.model_dump(by_alias=True)

    # Assert
    assert "averagePrice" in data
    assert "minPrice" in data
    assert "maxPrice" in data
