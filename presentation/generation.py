from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from application.generation.commands import GenerateTextCommand, GenerateTextCommandHandler

from .requests import TextGenerationRequest

router = APIRouter(prefix="/generation")

@router.post("/text")
def text(request: TextGenerationRequest):
    print("Generating text...")
    model_name = "TheBloke/LLaMA2-13B-Tiefighter-GPTQ"
    command = GenerateTextCommand(model_name, request.messages, request.human_name, request.ai_name)
    command_handler = GenerateTextCommandHandler(command)
    return StreamingResponse(command_handler.generate_text(), media_type="text/plain")
