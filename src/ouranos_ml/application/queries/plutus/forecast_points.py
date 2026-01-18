from ouranos_ml.domain.plutus.forecast_point import PlutusForecastPoint
from ouranos_ml.infrastructure.plutus.forecast_generator import ForecastGenerator


def forecast_points(sequences: list[list[PlutusForecastPoint]], numPredictions: int) -> list[list[PlutusForecastPoint]]:
    """Forecast future points for multiple sequences based on historical data.

    :param sequences: List of sequences, where each sequence contains historical forecast points.
    :param numPredictions: Number of future points to forecast.
    :return: List of lists containing forecasted points for each sequence.
    """
    generator = ForecastGenerator()
    all_predictions: list[list[PlutusForecastPoint]] = [[] for _ in sequences]
    current_sequences = [seq.copy() for seq in sequences]

    for _ in range(numPredictions):
        next_points = generator.predict_next(current_sequences)
        for i, (next_point, sequence) in enumerate(zip(next_points, current_sequences, strict=False)):
            sequence.append(next_point)
            sequence.pop(0)
            all_predictions[i].append(next_point)

    return all_predictions
