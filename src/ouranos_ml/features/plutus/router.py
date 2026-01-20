from fastapi import APIRouter

plutus_router = APIRouter(prefix="/plutus")

# Routes
from ouranos_ml.features.plutus.forecast.api import forecast
