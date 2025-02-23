from src.domain.common.base_schema import BaseSchema

from src.domain.plutus.forecast_point import PlutusForecastPoint

class TextGenerationRequest(BaseSchema):
    messages: list[dict[str, str]]

class PlutusForecastingRequest(BaseSchema):
    points: list[PlutusForecastPoint]
    num_predictions: int
