from fastapi import APIRouter

from src.api.schemas.requests import PlutusForecastingRequest
from src.application.queries.plutus.forecast_points import forecast_points

router = APIRouter(prefix="/plutus")

@router.post("/forecast")
def forecast(request: PlutusForecastingRequest):
    print("Forecasting Plutus data...")
    return forecast_points(request.points, request.num_predictions)
