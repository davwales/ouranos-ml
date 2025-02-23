from src.domain.common.base_schema import BaseSchema

class PlutusForecastPoint(BaseSchema):
    average_price: float
    min_price: float
    max_price: float
    volume: float
