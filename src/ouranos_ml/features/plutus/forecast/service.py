import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import structlog
import torch

from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint
from ouranos_ml.shared.inference.harness import Harness
from ouranos_ml.shared.inference.model import Model

logger = structlog.get_logger(__name__)

_DEFAULT_FEATURE_FIELDS = ["average_price", "min_price", "max_price", "volume"]
_DEFAULT_SEQUENCE_LENGTH = 30


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

    def predict_next(self, sequences: list[list[PlutusForecastPoint]]) -> list[PlutusForecastPoint]:
        """Predicts the next point for multiple sequences based on historical data."""
        if not all(len(seq) == self.sequence_length for seq in sequences):
            invalid_lengths = [i for i, seq in enumerate(sequences) if len(seq) != self.sequence_length]
            logger.warning(
                "invalid forecast sequence lengths",
                invalid_indices=invalid_lengths,
                expected_length=self.sequence_length,
            )
            raise ValueError(
                f"All sequences must have {self.sequence_length} points. Invalid sequences at indices: {invalid_lengths}"
            )

        batch_sequences = np.array(
            [[[getattr(p, field) for field in self.feature_fields] for p in sequence] for sequence in sequences]
        )

        scales = np.maximum(np.max(batch_sequences, axis=1, keepdims=True), 1e-8)
        normalized_sequences = batch_sequences / scales
        predictions = self.harness.predict(torch.as_tensor(normalized_sequences, dtype=torch.float32))
        denormalized_predictions = predictions[:, 0, :] * scales[:, 0, :]

        return [
            PlutusForecastPoint(
                average_price=pred[0],
                min_price=pred[1],
                max_price=pred[2],
                volume=pred[3],
            )
            for pred in denormalized_predictions
        ]


@lru_cache
def _get_forecast_generator() -> ForecastGenerator:
    """Build the forecast generator once per process.

    The trained artifact is loaded on the first forecast request; restart the
    service to pick up a retrained model.
    """
    return ForecastGenerator()


def forecast_points(
    sequences: list[list[PlutusForecastPoint]], num_predictions: int
) -> list[list[PlutusForecastPoint]]:
    """Forecast future points for multiple sequences based on historical data."""
    logger.info("forecast requested", sequence_count=len(sequences), num_predictions=num_predictions)
    generator = _get_forecast_generator()
    all_predictions: list[list[PlutusForecastPoint]] = [[] for _ in sequences]
    current_sequences = [seq.copy() for seq in sequences]

    for _ in range(num_predictions):
        next_points = generator.predict_next(current_sequences)
        for i, (next_point, sequence) in enumerate(zip(next_points, current_sequences, strict=True)):
            sequence.append(next_point)
            sequence.pop(0)
            all_predictions[i].append(next_point)

    logger.debug(
        "forecast complete",
        sequence_count=len(sequences),
        prediction_count=len(sequences) * num_predictions,
    )
    return all_predictions
