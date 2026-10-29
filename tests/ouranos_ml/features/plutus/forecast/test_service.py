import json
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ouranos_ml.features.plutus.forecast.service import (
    ForecastGenerator,
    SequenceLengthError,
    forecast_points,
    get_forecast_generator,
)
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint
from tests.ouranos_ml.shared.factories.forecast_factories import make_forecast_point, make_sequence
from tests.ouranos_ml.shared.otel_mocks import patch_operation_telemetry

SERVICE = "ouranos_ml.features.plutus.forecast.service"


def _generator_with_mock_harness(mock_harness: MagicMock) -> ForecastGenerator:
    """Build a ForecastGenerator that skips artifact loading via __new__."""
    generator = ForecastGenerator.__new__(ForecastGenerator)
    generator.harness = mock_harness
    generator.sequence_length = 30
    generator.feature_fields = ["average_price", "min_price", "max_price", "volume"]
    return generator


def test_predict_next_when_valid_30_point_sequence_should_return_prediction():
    # Arrange
    uniform_point = PlutusForecastPoint(average_price=10.0, min_price=10.0, max_price=10.0, volume=10.0)
    points = [uniform_point] * 30
    mock_harness = MagicMock()
    mock_harness.predict.return_value = np.array([[[2.0, 3.0, 4.0, 5.0]]])
    generator = _generator_with_mock_harness(mock_harness)

    # Act
    result = generator.predict_next([points])

    # Assert
    assert len(result) == 1
    assert isinstance(result[0], PlutusForecastPoint)
    assert result[0].average_price == 20.0
    assert result[0].min_price == 30.0
    assert result[0].max_price == 40.0
    assert result[0].volume == 50.0


def test_predict_next_when_feature_fields_reordered_should_map_predictions_by_field():
    # Arrange
    uniform_point = PlutusForecastPoint(average_price=10.0, min_price=10.0, max_price=10.0, volume=10.0)
    points = [uniform_point] * 30
    mock_harness = MagicMock()
    mock_harness.predict.return_value = np.array([[[2.0, 3.0, 4.0, 5.0]]])
    generator = _generator_with_mock_harness(mock_harness)
    generator.feature_fields = ["volume", "max_price", "average_price", "min_price"]

    # Act
    result = generator.predict_next([points])

    # Assert
    assert result[0].volume == 20.0
    assert result[0].max_price == 30.0
    assert result[0].average_price == 40.0
    assert result[0].min_price == 50.0


@pytest.mark.parametrize(
    "feature_fields",
    [
        ["average_price", "min_price", "max_price"],
        ["average_price", "min_price", "max_price", "volume", "volume"],
        ["average_price", "min_price", "max_price", "close_price"],
    ],
)
def test_forecast_generator_init_when_feature_fields_invalid_should_raise_value_error(tmp_path, feature_fields):
    # Arrange
    model_dir = tmp_path / "plutus-test"
    model_dir.mkdir()
    params = {"hidden_size": 8, "num_layers": 1, "dropout": 0.0, "feature_fields": feature_fields}
    (model_dir / "params.json").write_text(json.dumps(params))
    settings = MagicMock(models_dir=str(tmp_path), plutus_forecast_model_name="plutus-test")

    # Act & Assert
    with (
        patch("ouranos_ml.features.plutus.forecast.service.get_settings", return_value=settings),
        patch("ouranos_ml.features.plutus.forecast.service.Model") as mock_model,
        pytest.raises(ValueError, match="feature_fields"),
    ):
        ForecastGenerator()
    mock_model.assert_not_called()


def test_predict_next_when_feature_max_is_zero_should_not_produce_nan():
    # Arrange
    zero_point = PlutusForecastPoint(average_price=10.0, min_price=10.0, max_price=10.0, volume=0.0)
    points = [zero_point] * 30
    mock_harness = MagicMock()
    mock_harness.predict.return_value = np.array([[[2.0, 3.0, 4.0, 5.0]]])
    generator = _generator_with_mock_harness(mock_harness)

    # Act
    result = generator.predict_next([points])

    # Assert
    assert math.isfinite(result[0].average_price)
    assert math.isfinite(result[0].volume)
    assert result[0].average_price == pytest.approx(20.0)
    assert result[0].volume == pytest.approx(5.0 * 1e-8)


def test_forecast_points_when_valid_sequences_should_return_predictions():
    # Arrange
    mock_generator = MagicMock(spec=ForecastGenerator)
    mock_generator.predict_next.return_value = [make_forecast_point(average_price=42.0)]

    # Act
    with patch("ouranos_ml.features.plutus.forecast.service.get_forecast_generator", return_value=mock_generator):
        points = make_sequence(30)
        result = forecast_points([points], 3)

    # Assert
    assert len(result) == 1
    assert len(result[0]) == 3
    assert result[0][0].average_price == 42.0


def test_predict_next_when_sequence_not_30_points_should_raise_sequence_length_error():
    # Arrange
    uniform_point = PlutusForecastPoint(average_price=10.0, min_price=10.0, max_price=10.0, volume=10.0)
    short_sequence = [uniform_point] * 20
    generator = ForecastGenerator.__new__(ForecastGenerator)
    generator.sequence_length = 30

    # Act & Assert
    with pytest.raises(SequenceLengthError, match="30 points"):
        generator.predict_next([short_sequence])


def test_predict_next_when_some_sequences_invalid_should_report_their_indices():
    # Arrange
    generator = ForecastGenerator.__new__(ForecastGenerator)
    generator.sequence_length = 30
    sequences = [make_sequence(30), make_sequence(20), make_sequence(30), make_sequence(31)]

    # Act
    with pytest.raises(SequenceLengthError) as exc_info:
        generator.predict_next(sequences)

    # Assert
    assert exc_info.value.invalid_indices == [1, 3]
    assert exc_info.value.expected_length == 30


def test_sequence_length_error_when_raised_should_be_value_error():
    # Arrange
    error = SequenceLengthError([0], 30)

    # Act
    is_value_error = isinstance(error, ValueError)

    # Assert
    assert is_value_error
    assert str(error) == "All sequences must have 30 points. Invalid sequences at indices: [0]"


def test_forecast_points_when_sequence_not_30_points_should_raise_sequence_length_error():
    # Arrange
    mock_generator = MagicMock(spec=ForecastGenerator)
    mock_generator.predict_next.side_effect = SequenceLengthError([0], 30)

    # Act & Assert
    with patch("ouranos_ml.features.plutus.forecast.service.get_forecast_generator", return_value=mock_generator):
        points = make_sequence(20)
        with pytest.raises(SequenceLengthError, match="30 points"):
            forecast_points([points], 3)


def test_forecast_points_when_multiple_sequences_should_return_per_sequence_predictions():
    # Arrange
    mock_generator = MagicMock(spec=ForecastGenerator)
    mock_generator.predict_next.return_value = [
        make_forecast_point(average_price=42.0),
        make_forecast_point(average_price=99.0),
    ]

    # Act
    with patch("ouranos_ml.features.plutus.forecast.service.get_forecast_generator", return_value=mock_generator):
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
    with patch("ouranos_ml.features.plutus.forecast.service.get_forecast_generator", return_value=mock_generator):
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
    with patch("ouranos_ml.features.plutus.forecast.service.get_forecast_generator", return_value=mock_generator):
        points = make_sequence(30)
        forecast_points([points], 3)

    # Assert
    assert all(length == 30 for length in sequence_lengths)


def testget_forecast_generator_when_called_twice_should_return_same_instance():
    # Arrange
    get_forecast_generator.cache_clear()

    # Act
    first = get_forecast_generator()
    second = get_forecast_generator()

    # Assert
    assert first is second

    get_forecast_generator.cache_clear()


def test_forecast_points_when_called_should_record_forecast_span():
    # Arrange
    mock_generator = MagicMock(spec=ForecastGenerator)
    mock_generator.predict_next.return_value = [make_forecast_point(), make_forecast_point()]

    # Act
    with (
        patch_operation_telemetry() as telemetry,
        patch(f"{SERVICE}.get_forecast_generator", return_value=mock_generator),
    ):
        forecast_points([make_sequence(30), make_sequence(30)], 3)

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.name == "plutus.forecast"
    assert dict(span.attributes) == {"plutus.sequence_count": 2, "plutus.num_predictions": 3, "outcome": "completed"}
    telemetry.durations["plutus.forecast"].record.assert_called_once()


def test_get_forecast_generator_when_loading_should_record_load_span_once():
    # Arrange
    get_forecast_generator.cache_clear()

    # Act
    with patch_operation_telemetry() as telemetry, patch(f"{SERVICE}.ForecastGenerator"):
        get_forecast_generator()
        get_forecast_generator()
    get_forecast_generator.cache_clear()

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.name == "plutus.load_model"
    assert span.attributes["plutus.model"] == "plutus-forecasting-v1"
    duration = telemetry.durations["plutus.load_model"].record
    duration.assert_called_once()
    assert duration.call_args.args[1] == {"plutus.model": "plutus-forecasting-v1", "outcome": "completed"}
