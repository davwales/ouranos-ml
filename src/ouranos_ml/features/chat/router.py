from fastapi import APIRouter

chat_router = APIRouter(prefix="/chat")

# Routes
from ouranos_ml.features.chat.completions.api import completions
