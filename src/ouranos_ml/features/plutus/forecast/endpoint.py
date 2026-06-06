from fastapi import APIRouter

from ouranos_ml.features.plutus.forecast.schemas import ForecastRequest
from ouranos_ml.features.plutus.forecast.service import forecast_points
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint


def _forecast(request: ForecastRequest) -> list[list[PlutusForecastPoint]]:
    """Predicts future datapoints given a sequence of prior Plutus datapoints.
    Predictions are made using the model trained from the 'plutus_forecasting' experiment.
    """
    return forecast_points(request.points, request.num_predictions)


def register(router: APIRouter) -> None:
    """Register Plutus forecasting endpoints on the provided router."""
    router.post("/forecast")(_forecast)