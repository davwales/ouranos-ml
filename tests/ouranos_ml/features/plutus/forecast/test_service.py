from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ouranos_ml.features.plutus.forecast.service import ForecastGenerator, forecast_points
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint
from tests.ouranos_ml.shared.factories.forecast_factories import make_forecast_point, make_sequence


def test_predict_next_when_valid_30_point_sequence_should_return_prediction():
    # Arrange
    uniform_point = PlutusForecastPoint(average_price=10.0, min_price=10.0, max_price=10.0, volume=10.0)
    points = [uniform_point] * 30
    mock_harness = MagicMock()
    mock_harness.predict.return_value = np.array([[[2.0, 3.0, 4.0, 5.0]]])
    generator = ForecastGenerator.__new__(ForecastGenerator)
    generator.harness = mock_harness

    # Act
    result = generator.predict_next([points])

    # Assert
    assert len(result) == 1
    assert isinstance(result[0], PlutusForecastPoint)
    assert result[0].average_price == 20.0
    assert result[0].min_price == 30.0
    assert result[0].max_price == 40.0
    assert result[0].volume == 50.0


def test_forecast_points_when_valid_sequences_should_return_predictions():
    # Arrange
    mock_generator = MagicMock(spec=ForecastGenerator)
    mock_generator.predict_next.return_value = [make_forecast_point(average_price=42.0)]

    # Act
    with patch("ouranos_ml.features.plutus.forecast.service.ForecastGenerator", return_value=mock_generator):
        points = make_sequence(30)
        result = forecast_points([points], 3)

    # Assert
    assert len(result) == 1
    assert len(result[0]) == 3
    assert result[0][0].average_price == 42.0


def test_forecast_points_when_sequence_not_30_points_should_raise_value_error():
    # Arrange
    mock_generator = MagicMock(spec=ForecastGenerator)
    mock_generator.predict_next.side_effect = ValueError(
        "All sequences must have 30 points. Invalid sequences at indices: [0]"
    )

    # Act & Assert
    with patch("ouranos_ml.features.plutus.forecast.service.ForecastGenerator", return_value=mock_generator):
        points = make_sequence(20)
        with pytest.raises(ValueError, match="30 points"):
            forecast_points([points], 3)


def test_forecast_points_when_multiple_sequences_should_return_per_sequence_predictions():
    # Arrange
    mock_generator = MagicMock(spec=ForecastGenerator)
    mock_generator.predict_next.return_value = [
        make_forecast_point(average_price=42.0),
        make_forecast_point(average_price=99.0),
    ]

    # Act
    with patch("ouranos_ml.features.plutus.forecast.service.ForecastGenerator", return_value=mock_generator):
        seq1 = make_sequence(30)
        seq2 = make_sequence(30)
        result = forecast_points([seq1, seq2], 2)

    # Assert
    assert len(result) == 2
    assert len(result[0]) == 2
    assert len(result[1]) == 2
    assert result[0][0].average_price == 42.0
    assert result[1][0].average_price == 99.0


def test_forecast_points_when_num_predictions_zero_should_return_empty_lists():
    # Arrange
    mock_generator = MagicMock(spec=ForecastGenerator)

    # Act
    with patch("ouranos_ml.features.plutus.forecast.service.ForecastGenerator", return_value=mock_generator):
        points = make_sequence(30)
        result = forecast_points([points], 0)

    # Assert
    assert len(result) == 1
    assert result[0] == []
    mock_generator.predict_next.assert_not_called()


def test_forecast_points_when_sliding_window_should_maintain_30_point_sequences():
    # Arrange
    sequence_lengths = []
    mock_generator = MagicMock(spec=ForecastGenerator)

    def track_sequence_lengths(sequences):
        sequence_lengths.extend(len(seq) for seq in sequences)
        return [make_forecast_point(average_price=5.0)]

    mock_generator.predict_next.side_effect = track_sequence_lengths

    # Act
    with patch("ouranos_ml.features.plutus.forecast.service.ForecastGenerator", return_value=mock_generator):
        points = make_sequence(30)
        forecast_points([points], 3)

    # Assert
    assert all(length == 30 for length in sequence_lengths)
