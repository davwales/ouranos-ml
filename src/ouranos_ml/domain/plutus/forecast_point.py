from ouranos_ml.domain.common.base_schema import BaseSchema


class PlutusForecastPoint(BaseSchema):
    """Represents a single Plutus datapoint in the context of forecasting."""

    average_price: float
    min_price: float
    max_price: float
    volume: float
