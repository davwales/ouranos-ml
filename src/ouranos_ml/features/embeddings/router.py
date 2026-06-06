from fastapi import APIRouter

from ouranos_ml.features.embeddings.create_embeddings.endpoint import register as register_create_embeddings

embeddings_router = APIRouter(prefix="/embeddings")
register_create_embeddings(embeddings_router)