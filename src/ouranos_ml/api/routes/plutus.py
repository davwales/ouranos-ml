from fastapi import APIRouter

from ouranos_ml.api.schemas.requests import PlutusForecastingRequest
from ouranos_ml.application.queries.plutus.forecast_points import forecast_points
from ouranos_ml.domain.plutus.forecast_point import PlutusForecastPoint

router = APIRouter(prefix="/plutus")


@router.post("/forecast")
def forecast(request: PlutusForecastingRequest) -> list[list[PlutusForecastPoint]]:
    """Predicts future datapoints given a sequence of prior Plutus datapoints.
    Predicitons are made using the model trained from the 'plutus_forecasting' experiment.
    """
    return forecast_points(request.points, request.num_predictions)
