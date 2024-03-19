from pydantic import BaseModel
from domain.generation import GeneratorType

class ImageGenerationRequest(BaseModel):
    type: GeneratorType
    file_name: str | None = None
    prompt: str
    height: int
    width: int
    negative_prompt: str | None = None
    num_inference_steps: int = 20
    seed: int = None
