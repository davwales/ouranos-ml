import copy
import json
from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import structlog
import torch

from ouranos_ml.features.plutus.forecast.schemas import MAX_FORECAST_SEQUENCES, ForecastBand
from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint
from ouranos_ml.shared.inference.harness import Harness
from ouranos_ml.shared.inference.model import Model
from ouranos_ml.shared.telemetry import start_operation

logger = structlog.get_logger(__name__)

_DEFAULT_FEATURE_FIELDS = ["average_price", "min_price", "max_price", "volume"]
_DEFAULT_SEQUENCE_LENGTH = 30
_BAND_QUANTILES = (0.1, 0.5, 0.9)


class SequenceLengthError(ValueError):
    """Raised when one or more sequences do not have the model's expected number of points."""

    def __init__(self, invalid_indices: list[int], expected_length: int) -> None:
        self.invalid_indices = invalid_indices
        self.expected_length = expected_length
        super().__init__(
            f"All sequences must have {expected_length} points. Invalid sequences at indices: {invalid_indices}"
        )


class ForecastGenerator:
    """Generator used to predict future Plutus datapoints using the model trained as part of the 'plutus_forecasting' experiment."""

    def __init__(self) -> None:
        settings = get_settings()
        model_path = Path(settings.models_dir) / settings.plutus_forecast_model_name
        logger.info("loading forecast model", model_path=str(model_path))
        params_file = model_path / "params.json"

        try:
            with open(params_file) as f:
                params = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("forecast model params load failed", model_path=str(params_file), exc_info=exc)
            raise

        self.sequence_length = params.get("sequence_length", _DEFAULT_SEQUENCE_LENGTH)
        self.feature_fields = params.get("feature_fields", _DEFAULT_FEATURE_FIELDS)
        expected_fields = set(PlutusForecastPoint.model_fields)
        if len(self.feature_fields) != len(expected_fields) or set(self.feature_fields) != expected_fields:
            logger.error("invalid forecast feature fields", feature_fields=self.feature_fields)
            raise ValueError(
                f"feature_fields must list each of {sorted(expected_fields)} exactly once, got {self.feature_fields}"
            )
        model = Model(
            input_size=params.get("input_size", len(_DEFAULT_FEATURE_FIELDS)),
            output_size=params.get("output_size", len(_DEFAULT_FEATURE_FIELDS)),
            hidden_size=params["hidden_size"],
            prediction_horizon=params.get("prediction_horizon", 1),
            num_layers=params["num_layers"],
            dropout=params["dropout"],
        )
        self.harness = Harness(model)
        self.harness.load_model(str(model_path / "model.pth"))
        self.harness.model.eval()
        # Monte Carlo dropout: a copy left in training mode keeps dropout active (the model has no batch norm). It is a
        # copy because switching the shared model's mode would leak into concurrent requests.
        self._sampling_model = copy.deepcopy(self.harness.model).train()

    def forecast(
        self, sequences: list[list[PlutusForecastPoint]], num_predictions: int
    ) -> list[list[PlutusForecastPoint]]:
        """Forecasts the next ``num_predictions`` points of each sequence."""
        self._check_lengths(sequences)
        predictions = self._roll(self.harness.model, self._to_tensor(sequences), num_predictions)
        return self._to_points(predictions.numpy())

    def forecast_bands(
        self, sequences: list[list[PlutusForecastPoint]], num_predictions: int, samples: int
    ) -> list[list[ForecastBand]]:
        """Forecasts p10/p50/p90 bands for each sequence from ``samples`` Monte Carlo dropout trajectories."""
        self._check_lengths(sequences)
        windows = self._to_tensor(sequences).repeat(samples, 1, 1)
        # Chunked so GPU memory is bounded by the chunk size, however many samples are drawn. repeat() is sample-major,
        # so the trajectories reshape to (samples, sequences, num_predictions, feature).
        trajectories = torch.cat(
            [
                self._roll(self._sampling_model, chunk, num_predictions)
                for chunk in windows.split(MAX_FORECAST_SEQUENCES)
            ]
        ).view(samples, len(sequences), num_predictions, -1)
        lower, median, upper = (self._to_points(q) for q in np.quantile(trajectories.numpy(), _BAND_QUANTILES, axis=0))
        return [
            [ForecastBand(lower=low, median=mid, upper=high) for low, mid, high in zip(*bands, strict=True)]
            for bands in zip(lower, median, upper, strict=True)
        ]

    def _check_lengths(self, sequences: list[list[PlutusForecastPoint]]) -> None:
        """Raises a SequenceLengthError listing every sequence that isn't ``sequence_length`` points long."""
        invalid_indices = [i for i, sequence in enumerate(sequences) if len(sequence) != self.sequence_length]
        if invalid_indices:
            logger.warning(
                "invalid forecast sequence lengths",
                invalid_indices=invalid_indices,
                expected_length=self.sequence_length,
            )
            raise SequenceLengthError(invalid_indices, self.sequence_length)

    def _to_tensor(self, sequences: list[list[PlutusForecastPoint]]) -> torch.Tensor:
        """Converts the sequences to a ``(batch, sequence_length, feature)`` tensor on the model's device."""
        values = [[[getattr(point, field) for field in self.feature_fields] for point in seq] for seq in sequences]
        # float64 keeps the denormalized values exact; the model itself runs in float32.
        return torch.tensor(values, dtype=torch.float64, device=self.harness.device)

    def _to_points(self, predictions: np.ndarray) -> list[list[PlutusForecastPoint]]:
        """Maps a ``(batch, num_predictions, feature)`` array back to points by ``feature_fields``."""
        return [
            [PlutusForecastPoint(**dict(zip(self.feature_fields, values, strict=True))) for values in sequence]
            for sequence in predictions.tolist()
        ]

    @staticmethod
    def _roll(model: torch.nn.Module, windows: torch.Tensor, num_predictions: int) -> torch.Tensor:
        """Predicts one step at a time, sliding each prediction into its window.

        Every step is one forward pass over the whole batch. Windows are max-scaled per sequence and feature before
        each step, matching the experiment's ``sequence_transform``. Only the first step of the model's horizon is used.
        """
        batch_size, _, feature_count = windows.shape
        predictions = windows.new_empty((batch_size, num_predictions, feature_count))
        with torch.inference_mode():
            for step in range(num_predictions):
                scales = windows.amax(dim=1, keepdim=True).clamp_min(1e-8)
                next_points = model((windows / scales).float())[:, :1, :] * scales
                predictions[:, step] = next_points[:, 0]
                windows = torch.cat((windows[:, 1:], next_points), dim=1)
        return predictions.cpu()


@lru_cache
def get_forecast_generator() -> ForecastGenerator:
    """Build the forecast generator once per process.

    The trained artifact is loaded on the first forecast request or health probe; a failed load is not cached,
    so the next call retries. Restart the service to pick up a retrained model.
    """
    with start_operation(
        "plutus.load_model", metric_attributes={"plutus.model": get_settings().plutus_forecast_model_name}
    ):
        return ForecastGenerator()


def forecast_points(
    sequences: list[list[PlutusForecastPoint]], num_predictions: int
) -> list[list[PlutusForecastPoint]]:
    """Forecast future points for multiple sequences based on historical data."""
    with _forecast_operation(sequences, num_predictions):
        return get_forecast_generator().forecast(sequences, num_predictions)


def forecast_bands(
    sequences: list[list[PlutusForecastPoint]], num_predictions: int, samples: int
) -> list[list[ForecastBand]]:
    """Forecast p10/p50/p90 bands for multiple sequences based on historical data."""
    with _forecast_operation(sequences, num_predictions, samples):
        return get_forecast_generator().forecast_bands(sequences, num_predictions, samples)


@contextmanager
def _forecast_operation(
    sequences: list[list[PlutusForecastPoint]], num_predictions: int, samples: int | None = None
) -> Generator[None]:
    """Log and trace one forecast request."""
    logger.info("forecast requested", sequence_count=len(sequences), num_predictions=num_predictions, samples=samples)
    attributes = {"plutus.sequence_count": len(sequences), "plutus.num_predictions": num_predictions}
    if samples is not None:
        attributes["plutus.samples"] = samples
    with start_operation("plutus.forecast", attributes=attributes):
        yield

    logger.debug(
        "forecast complete",
        sequence_count=len(sequences),
        prediction_count=len(sequences) * num_predictions,
    )
