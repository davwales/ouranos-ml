from src.domain.plutus.forecast_point import PlutusForecastPoint
from src.infrastructure.plutus.forecast_generator import ForecastGenerator

def forecast_points(points: list[PlutusForecastPoint], numPredictions: int) -> list[PlutusForecastPoint]:
    """
    Forecast future points based on historical data.

    :param points: List of historical forecast points.
    :param numPredictions: Number of future points to forecast.
    :return: List of forecasted points.
    """
    generator = ForecastGenerator()
    predictions = []
    for i in range(numPredictions):
        next_point = generator.predict_next(points)
        points.append(next_point)
        points.remove(points[0])
        predictions.append(next_point)
    return predictions
