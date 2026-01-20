from ouranos_ml.features.plutus.forecast.integration import ForecastGenerator
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint


def forecast_points(sequences: list[list[PlutusForecastPoint]], numPredictions: int) -> list[list[PlutusForecastPoint]]:
    """Forecast future points for multiple sequences based on historical data."""
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
