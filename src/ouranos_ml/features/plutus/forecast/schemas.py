from ouranos_ml.shared.domain.core.base_schema import BaseSchema
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint
from pydantic import Field


class ForecastRequest(BaseSchema):
    """Request for predicting future Plutus datapoints."""

    points: list[list[PlutusForecastPoint]]
    num_predictions: int = Field(ge=1, le=500)
