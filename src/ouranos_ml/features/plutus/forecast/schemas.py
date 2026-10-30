from typing import Annotated

from ouranos_ml.shared.domain.core.base_schema import BaseSchema
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint
from pydantic import Field

# Pantheon batches up to 500 symbols per request (Plutus:Forecasting:BatchSize); keep headroom above that.
MAX_FORECAST_SEQUENCES = 1000


class ForecastRequest(BaseSchema):
    """Request for predicting future Plutus datapoints."""

    points: list[Annotated[list[PlutusForecastPoint], Field(min_length=1)]] = Field(
        min_length=1, max_length=MAX_FORECAST_SEQUENCES
    )
    num_predictions: int = Field(ge=1, le=500)
