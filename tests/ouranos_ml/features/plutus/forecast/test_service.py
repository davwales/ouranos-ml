import copy
import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ouranos_ml.features.plutus.forecast.schemas import MAX_FORECAST_SEQUENCES, MAX_NUM_PREDICTIONS
from ouranos_ml.features.plutus.forecast.service import (
    ForecastGenerator,
    SequenceLengthError,
    forecast_bands,
    forecast_points,
    get_forecast_generator,
)
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint
from tests.conftest import TORCH_AVAILABLE
from tests.ouranos_ml.shared.factories.forecast_factories import (
    make_forecast_band,
    make_forecast_point,
    make_sequence,
)
from tests.ouranos_ml.shared.otel_mocks import patch_operation_telemetry

if TORCH_AVAILABLE:
    import torch

    from ouranos_ml.shared.inference.harness import Harness
    from ouranos_ml.shared.inference.model import Model

SERVICE = "ouranos_ml.features.plutus.forecast.service"
FEATURE_FIELDS = ["average_price", "min_price", "max_price", "volume"]

requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="real torch is not installed")


def _generator(
    model: "torch.nn.Module",
    feature_fields: list[str] = FEATURE_FIELDS,
    sampling_model: "torch.nn.Module | None" = None,
) -> ForecastGenerator:
    """Build a 30-point ForecastGenerator around ``model`` on the CPU, skipping artifact loading via __new__."""
    generator = ForecastGenerator.__new__(ForecastGenerator)
    generator.harness = MagicMock(model=model, device=torch.device("cpu"))
    generator._sampling_model = sampling_model
    generator.sequence_length = 30
    generator.feature_fields = feature_fields
    return generator


def _constant_model(prediction: list[float]) -> MagicMock:
    """A model that predicts the same normalized point for every window."""
    model = MagicMock()
    model.side_effect = lambda windows: torch.tensor(prediction).expand(windows.shape[0], 1, len(prediction))
    return model


def _tiny_model() -> "Model":
    """A small, randomly initialised forecasting model in eval mode."""
    torch.manual_seed(0)
    return Model(input_size=4, hidden_size=8, prediction_horizon=1, output_size=4, num_layers=2, dropout=0.2).eval()


def _uniform_sequence(value: float, volume: float | None = None) -> list[PlutusForecastPoint]:
    """A 30-point sequence whose fields all equal ``value`` (volume optionally overridden)."""
    point = PlutusForecastPoint(
        average_price=value, min_price=value, max_price=value, volume=value if volume is None else volume
    )
    return [point] * 30


def _reference_forecast(model: "Model", windows: np.ndarray, num_predictions: int) -> np.ndarray:
    """The pre-vectorisation algorithm: rescale the window, predict and slide, one step at a time."""
    predictions = []
    for _ in range(num_predictions):
        scales = np.maximum(windows.max(axis=1, keepdims=True), 1e-8)
        with torch.no_grad():
            next_point = model(torch.as_tensor(windows / scales, dtype=torch.float32)).numpy()[:, :1, :] * scales
        predictions.append(next_point)
        windows = np.concatenate((windows[:, 1:], next_point), axis=1)
    return np.concatenate(predictions, axis=1)


@requires_torch
def test_forecast_when_uniform_sequence_should_denormalize_prediction():
    # Arrange
    generator = _generator(_constant_model([2.0, 3.0, 4.0, 5.0]))

    # Act
    result = generator.forecast([_uniform_sequence(10.0)], 1)

    # Assert
    assert result == [[PlutusForecastPoint(average_price=20.0, min_price=30.0, max_price=40.0, volume=50.0)]]


@requires_torch
def test_forecast_when_feature_fields_reordered_should_map_predictions_by_field():
    # Arrange
    generator = _generator(
        _constant_model([2.0, 3.0, 4.0, 5.0]), feature_fields=["volume", "max_price", "average_price", "min_price"]
    )

    # Act
    result = generator.forecast([_uniform_sequence(10.0)], 1)

    # Assert
    assert result == [[PlutusForecastPoint(volume=20.0, max_price=30.0, average_price=40.0, min_price=50.0)]]


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


@requires_torch
def test_forecast_when_feature_max_is_zero_should_not_produce_nan():
    # Arrange
    generator = _generator(_constant_model([2.0, 3.0, 4.0, 5.0]))

    # Act
    result = generator.forecast([_uniform_sequence(10.0, volume=0.0)], 1)

    # Assert
    assert result[0][0].average_price == pytest.approx(20.0)
    assert result[0][0].volume == pytest.approx(5.0 * 1e-8)


@requires_torch
def test_forecast_when_multiple_sequences_should_scale_each_sequence_independently():
    # Arrange
    generator = _generator(_constant_model([1.0, 1.0, 1.0, 1.0]))

    # Act
    result = generator.forecast([_uniform_sequence(10.0), _uniform_sequence(100.0)], 2)

    # Assert
    assert [[point.average_price for point in sequence] for sequence in result] == [[10.0, 10.0], [100.0, 100.0]]


@requires_torch
def test_forecast_when_multiple_steps_should_run_one_batched_pass_per_step_over_a_sliding_window():
    # Arrange
    model = _constant_model([2.0, 2.0, 2.0, 2.0])
    generator = _generator(model)

    # Act
    generator.forecast([_uniform_sequence(10.0)] * 3, 2)

    # Assert
    first_windows, second_windows = (call.args[0] for call in model.call_args_list)
    assert first_windows.shape == second_windows.shape == (3, 30, 4)
    # The first prediction (20) became the window's max, so it is the last row and everything before it halves.
    assert torch.equal(second_windows[:, -1], torch.ones(3, 4))
    assert torch.equal(second_windows[:, :-1], torch.full((3, 29, 4), 0.5))


@requires_torch
def test_forecast_when_real_model_should_match_step_by_step_reference():
    # Arrange
    model = _tiny_model()
    generator = _generator(model)
    sequences = [make_sequence(30), _uniform_sequence(250.0, volume=0.0), _uniform_sequence(3.0)]
    windows = np.array([[[getattr(p, field) for field in FEATURE_FIELDS] for p in seq] for seq in sequences])

    # Act
    result = generator.forecast(sequences, 5)

    # Assert
    expected = _reference_forecast(model, windows, 5)
    actual = np.array([[[getattr(p, field) for field in FEATURE_FIELDS] for p in seq] for seq in result])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


@requires_torch
def test_forecast_when_num_predictions_zero_should_return_empty_lists():
    # Arrange
    model = _constant_model([1.0, 1.0, 1.0, 1.0])
    generator = _generator(model)

    # Act
    result = generator.forecast([_uniform_sequence(10.0)], 0)

    # Assert
    assert result == [[]]
    model.assert_not_called()


@requires_torch
def test_forecast_when_max_request_size_should_finish_within_seconds():
    # Arrange
    generator = ForecastGenerator.__new__(ForecastGenerator)
    generator.harness = Harness(_tiny_model(), device=torch.device("cpu"))
    generator.sequence_length = 30
    generator.feature_fields = FEATURE_FIELDS
    sequences = [make_sequence(30)] * MAX_FORECAST_SEQUENCES

    # Act
    start = time.perf_counter()
    result = generator.forecast(sequences, MAX_NUM_PREDICTIONS)
    elapsed = time.perf_counter() - start

    # Assert
    assert len(result) == MAX_FORECAST_SEQUENCES
    assert len(result[0]) == MAX_NUM_PREDICTIONS
    # About 2s locally, against 15s for the per-point loop this replaced; the slack is for slower CI runners.
    assert elapsed < 10


@requires_torch
def test_forecast_generator_init_should_keep_a_training_mode_copy_for_sampling(tmp_path):
    # Arrange
    model_dir = tmp_path / "plutus-test"
    model_dir.mkdir()
    params = {"hidden_size": 8, "num_layers": 2, "dropout": 0.2}
    (model_dir / "params.json").write_text(json.dumps(params))
    torch.save(Model(input_size=4, output_size=4, prediction_horizon=1, **params).state_dict(), model_dir / "model.pth")
    settings = MagicMock(models_dir=str(tmp_path), plutus_forecast_model_name="plutus-test")

    # Act
    with patch(f"{SERVICE}.get_settings", return_value=settings):
        generator = ForecastGenerator()

    # Assert
    assert not generator.harness.model.training
    assert generator._sampling_model.training
    assert generator._sampling_model is not generator.harness.model
    for loaded, sampling in zip(
        generator.harness.model.parameters(), generator._sampling_model.parameters(), strict=True
    ):
        assert torch.equal(loaded, sampling)


@requires_torch
def test_forecast_bands_when_samples_vary_should_return_10th_50th_and_90th_percentiles():
    # Arrange
    sampling_model = MagicMock()
    # Sample i predicts i (normalized), so the 11 samples of a uniform-10 window are 0, 10, ..., 100.
    sampling_model.side_effect = lambda windows: (
        torch.arange(windows.shape[0], dtype=torch.float32).view(-1, 1, 1).expand(-1, 1, 4)
    )
    generator = _generator(_constant_model([1.0] * 4), sampling_model=sampling_model)

    # Act
    result = generator.forecast_bands([_uniform_sequence(10.0)], 1, samples=11)

    # Assert
    band = result[0][0]
    assert (band.lower.average_price, band.median.average_price, band.upper.average_price) == pytest.approx(
        (10.0, 50.0, 90.0)
    )


@requires_torch
def test_forecast_bands_when_samples_span_chunks_should_keep_each_sequence_s_samples_together():
    # Arrange
    sampling_model = _constant_model([1.0] * 4)
    generator = _generator(_constant_model([2.0] * 4), sampling_model=sampling_model)

    # Act
    with patch(f"{SERVICE}.MAX_FORECAST_SEQUENCES", 4):
        result = generator.forecast_bands([_uniform_sequence(10.0), _uniform_sequence(100.0)], 2, samples=3)

    # Assert
    assert [call.args[0].shape[0] for call in sampling_model.call_args_list] == [4, 4, 2, 2]
    assert [[band.median.average_price for band in sequence] for sequence in result] == [[10.0, 10.0], [100.0, 100.0]]
    assert all(band.lower == band.median == band.upper for sequence in result for band in sequence)


@requires_torch
def test_forecast_bands_when_real_model_should_spread_samples_around_median():
    # Arrange
    model = _tiny_model()
    generator = _generator(model, sampling_model=copy.deepcopy(model).train())

    # Act
    result = generator.forecast_bands([make_sequence(30)], 3, samples=50)

    # Assert
    for band in result[0]:
        for field in FEATURE_FIELDS:
            assert getattr(band.lower, field) <= getattr(band.median, field) <= getattr(band.upper, field)
    assert any(band.lower != band.upper for band in result[0])


def test_forecast_bands_when_sequence_not_30_points_should_raise_sequence_length_error():
    # Arrange
    generator = ForecastGenerator.__new__(ForecastGenerator)
    generator.sequence_length = 30

    # Act & Assert
    with pytest.raises(SequenceLengthError, match="30 points"):
        generator.forecast_bands([make_sequence(20)], 1, samples=10)


def test_forecast_when_sequence_not_30_points_should_raise_sequence_length_error():
    # Arrange
    generator = ForecastGenerator.__new__(ForecastGenerator)
    generator.sequence_length = 30

    # Act & Assert
    with pytest.raises(SequenceLengthError, match="30 points"):
        generator.forecast([make_sequence(20)], 1)


def test_forecast_when_some_sequences_invalid_should_report_their_indices():
    # Arrange
    generator = ForecastGenerator.__new__(ForecastGenerator)
    generator.sequence_length = 30
    sequences = [make_sequence(30), make_sequence(20), make_sequence(30), make_sequence(31)]

    # Act
    with pytest.raises(SequenceLengthError) as exc_info:
        generator.forecast(sequences, 1)

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


def test_forecast_points_when_valid_sequences_should_return_generator_forecast():
    # Arrange
    mock_generator = MagicMock(spec=ForecastGenerator)
    forecast = [[make_forecast_point(average_price=42.0)] * 3]
    mock_generator.forecast.return_value = forecast
    sequences = [make_sequence(30)]

    # Act
    with patch(f"{SERVICE}.get_forecast_generator", return_value=mock_generator):
        result = forecast_points(sequences, 3)

    # Assert
    assert result is forecast
    mock_generator.forecast.assert_called_once_with(sequences, 3)


def test_forecast_points_when_sequence_not_30_points_should_raise_sequence_length_error():
    # Arrange
    mock_generator = MagicMock(spec=ForecastGenerator)
    mock_generator.forecast.side_effect = SequenceLengthError([0], 30)

    # Act & Assert
    with (
        patch(f"{SERVICE}.get_forecast_generator", return_value=mock_generator),
        pytest.raises(SequenceLengthError, match="30 points"),
    ):
        forecast_points([make_sequence(20)], 3)


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
    mock_generator.forecast.return_value = [[make_forecast_point()] * 3] * 2

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


def test_forecast_bands_when_called_should_return_generator_bands_and_record_samples():
    # Arrange
    mock_generator = MagicMock(spec=ForecastGenerator)
    bands = [[make_forecast_band()] * 3]
    mock_generator.forecast_bands.return_value = bands
    sequences = [make_sequence(30)]

    # Act
    with (
        patch_operation_telemetry() as telemetry,
        patch(f"{SERVICE}.get_forecast_generator", return_value=mock_generator),
    ):
        result = forecast_bands(sequences, 3, 20)

    # Assert
    assert result is bands
    mock_generator.forecast_bands.assert_called_once_with(sequences, 3, 20)
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes["plutus.samples"] == 20


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
