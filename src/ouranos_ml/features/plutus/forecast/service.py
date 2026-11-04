import json
from functools import lru_cache
from pathlib import Path

import structlog
import torch

from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint
from ouranos_ml.shared.inference.harness import Harness
from ouranos_ml.shared.inference.model import Model
from ouranos_ml.shared.telemetry import start_operation

logger = structlog.get_logger(__name__)

_DEFAULT_FEATURE_FIELDS = ["average_price", "min_price", "max_price", "volume"]
_DEFAULT_SEQUENCE_LENGTH = 30


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

    def forecast(
        self, sequences: list[list[PlutusForecastPoint]], num_predictions: int
    ) -> list[list[PlutusForecastPoint]]:
        """Forecasts the next ``num_predictions`` points of each sequence."""
        self._check_lengths(sequences)
        windows = self._to_tensor(sequences)
        predictions = self._roll(windows, num_predictions)
        return [
            [PlutusForecastPoint(**dict(zip(self.feature_fields, values, strict=True))) for values in sequence]
            for sequence in predictions.tolist()
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

    def _roll(self, windows: torch.Tensor, num_predictions: int) -> torch.Tensor:
        """Predicts one step at a time, sliding each prediction into its window.

        Every step is one forward pass over the whole batch. Windows are max-scaled per sequence and feature before
        each step, matching the experiment's ``sequence_transform``. Only the first step of the model's horizon is used.
        """
        batch_size, _, feature_count = windows.shape
        predictions = windows.new_empty((batch_size, num_predictions, feature_count))
        with torch.inference_mode():
            for step in range(num_predictions):
                scales = windows.amax(dim=1, keepdim=True).clamp_min(1e-8)
                next_points = self.harness.model((windows / scales).float())[:, :1, :] * scales
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
    logger.info("forecast requested", sequence_count=len(sequences), num_predictions=num_predictions)
    attributes = {"plutus.sequence_count": len(sequences), "plutus.num_predictions": num_predictions}
    with start_operation("plutus.forecast", attributes=attributes):
        predictions = get_forecast_generator().forecast(sequences, num_predictions)

    logger.debug(
        "forecast complete",
        sequence_count=len(sequences),
        prediction_count=len(sequences) * num_predictions,
    )
    return predictions
