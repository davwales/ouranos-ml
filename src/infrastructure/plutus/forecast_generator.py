import numpy as np
import json
import torch

from src.domain.plutus.forecast_point import PlutusForecastPoint
from experiments.harness import Harness
from experiments.plutus.forecasting.model import Model

class ForecastGenerator:
    def __init__(self):
        experiment_path = 'experiments/plutus/forecasting'
        file = f'{experiment_path}/params.json'
        params = {}
        with open(file, 'r') as f:
            params = json.load(f)
            
        model = Model(
            input_size=4, 
            output_size=4,
            hidden_size=params['hidden_size'], 
            prediction_horizon=1,
            num_layers=params['num_layers'],
            dropout=params['dropout']
        )
        self.harness = Harness(model, None, None)
        self.harness.load_model(f'{experiment_path}/model.pth')

    def predict_next(self, points: list[PlutusForecastPoint]) -> PlutusForecastPoint:
        """
        Predicts the next point based on historical data.

        :param points: List of historical forecast points.
        :return: Prediction of the next forecast point.
        """
        if (len(points) != 30):
            raise ValueError(f"Expected 30 historical points, got '{len(points)}'.")

        sequence = np.array([[p.average_price, p.min_price, p.min_price, p.volume] for p in points])
        scale = np.max(sequence, axis=0)
        sequence = sequence / scale
        prediction = self.harness.predict(torch.FloatTensor([sequence]))[0][0]
        prediction = prediction * scale

        return PlutusForecastPoint(
            average_price=prediction[0],
            min_price=prediction[1],
            max_price=prediction[2],
            volume=prediction[3]
        )
