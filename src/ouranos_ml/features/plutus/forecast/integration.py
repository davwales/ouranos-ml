import json
from typing import Any

import numpy as np
import torch

from experiments.plutus_forecasting.model import Model
from experiments.utils.harness import Harness
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint


class ForecastGenerator:
    """Generator used to predict future Plutus datapoints using the model trained as part of the 'plutus_forecasting' experiment."""

    def __init__(self) -> None:
        experiment_path = "src/experiments/plutus_forecasting"
        file = f"{experiment_path}/params.json"
        params: dict[str, Any] = {}
        with open(file) as f:
            params = json.load(f)

        model = Model(
            input_size=4,
            output_size=4,
            hidden_size=params["hidden_size"],
            prediction_horizon=1,
            num_layers=params["num_layers"],
            dropout=params["dropout"],
        )
        self.harness = Harness(model)
        self.harness.load_model(f"{experiment_path}/model.pth")

    def predict_next(self, sequences: list[list[PlutusForecastPoint]]) -> list[PlutusForecastPoint]:
        """Predicts the next point for multiple sequences based on historical data."""
        if not all(len(seq) == 30 for seq in sequences):
            invalid_lengths = [i for i, seq in enumerate(sequences) if len(seq) != 30]
            raise ValueError(f"All sequences must have 30 points. Invalid sequences at indices: {invalid_lengths}")

        batch_sequences = np.array(
            [[[p.average_price, p.min_price, p.max_price, p.volume] for p in sequence] for sequence in sequences]
        )

        scales = np.max(batch_sequences, axis=1, keepdims=True)
        normalized_sequences = batch_sequences / scales
        predictions = self.harness.predict(torch.FloatTensor(normalized_sequences))
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
