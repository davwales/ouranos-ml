from ouranos_ml.domain.common.base_schema import BaseSchema
from ouranos_ml.domain.plutus.forecast_point import PlutusForecastPoint


class PlutusForecastingRequest(BaseSchema):
    """Request for predicting future Plutus datapoints."""

    points: list[list[PlutusForecastPoint]]
    num_predictions: int
