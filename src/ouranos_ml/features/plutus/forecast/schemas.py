from ouranos_ml.shared.domain.core.base_schema import BaseSchema
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint


class ForecastRequest(BaseSchema):
    """Request for predicting future Plutus datapoints."""

    points: list[list[PlutusForecastPoint]]
    num_predictions: int
