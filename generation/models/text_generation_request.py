from pydantic import BaseModel
from ..generator_type import GeneratorType

class TextGenerationRequest(BaseModel):
    type: GeneratorType
    context: list[str]
    instructions: list[str]
    response_start: str
    extra_stop_words: list[str] = []