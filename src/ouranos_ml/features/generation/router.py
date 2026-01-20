from fastapi import APIRouter

generation_router = APIRouter(prefix="/generation")

# Routes
from ouranos_ml.features.generation.chat.api import chat
