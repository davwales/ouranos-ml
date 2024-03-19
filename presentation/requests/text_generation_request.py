from pydantic import BaseModel
from domain.generation import GeneratorType

class TextGenerationRequest(BaseModel):
    type: GeneratorType
    context: list[str]
    instructions: list[str]
    response_start: str
    extra_stop_words: list[str] = []