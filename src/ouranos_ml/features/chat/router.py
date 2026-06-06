from fastapi import APIRouter

from ouranos_ml.features.chat.completions.endpoint import register as register_completions

chat_router = APIRouter(prefix="/chat")
register_completions(chat_router)