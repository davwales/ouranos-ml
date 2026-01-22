from fastapi import APIRouter

embeddings_router = APIRouter(prefix="/embeddings")

# Routes
from ouranos_ml.features.embeddings.create_embeddings.api import create_embeddings
