from fastapi import APIRouter

from ouranos_ml.features.plutus.forecast.endpoint import register as register_forecast

plutus_router = APIRouter(prefix="/plutus")
register_forecast(plutus_router)