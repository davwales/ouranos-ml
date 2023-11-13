from pydantic import BaseModel

class ImageGenerationResponse(BaseModel):
    seed: int
    file_path: str