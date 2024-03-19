import os
from fastapi import APIRouter
from fastapi_utils.tasks import repeat_every
from application.generation import GeneratorFactory
from domain.generation import GeneratorRegistration, GeneratorType
from infrastructure.hf import PipelineTextGenerator, ImageGenerator
from .requests import TextGenerationRequest, ImageGenerationRequest
from .responses import TextGenerationResponse, ImageGenerationResponse

router = APIRouter(prefix="/generation")

generator_factory = GeneratorFactory(12, [
    GeneratorRegistration(GeneratorType.TEXT_RP, 12, PipelineTextGenerator(
        model_name="TheBloke/LLaMA2-13B-Tiefighter-AWQ")),

    GeneratorRegistration(GeneratorType.IMAGE_REALISM, 9, ImageGenerator(
        checkpoint=os.path.relpath("nn_models/realisticVisionV51_v51VAE.safetensors"),
        upscaler="weights/RealESRGAN_x4.pth"
    ))
])

@router.post("/text")
def text(request: TextGenerationRequest):
    # print("Generating a completion...")
    generator: PipelineTextGenerator = generator_factory.get_generator(request.type)
    tokens, content = generator.generate(request.context, request.instructions, request.response_start, request.extra_stop_words)
    return TextGenerationResponse(type=request.type, tokens=tokens, content=content)

@router.post("/image")
def image(request: ImageGenerationRequest):
    # print("Generating an image...")
    generator: ImageGenerator = generator_factory.get_generator(request.type)
    seed, path = generator.generate(request.prompt, request.negative_prompt, request.width, request.height, request.num_inference_steps, request.seed, request.file_name)
    return ImageGenerationResponse(seed=seed, file_path=path)

@router.on_event("startup")
@repeat_every(seconds=60)
def purge_unused_generators():
    # print("Purging unused generators...")
    generator_factory.purge_unused(seconds=300)
