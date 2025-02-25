from src.domain.common.base_schema import BaseSchema
from src.domain.chat.chat_message import ChatMessage
from src.domain.plutus.forecast_point import PlutusForecastPoint

class TextGenerationRequest(BaseSchema):
    messages: list[ChatMessage]

class PlutusForecastingRequest(BaseSchema):
    points: list[list[PlutusForecastPoint]]
    num_predictions: int
