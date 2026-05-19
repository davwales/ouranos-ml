import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ouranos_ml.features.chat.router import chat_router
from ouranos_ml.features.embeddings.router import embeddings_router
from ouranos_ml.features.health.router import health_router
from ouranos_ml.features.models.router import models_router
from ouranos_ml.features.plutus.router import plutus_router
from ouranos_ml.shared.domain.core.settings import get_settings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,  # ty: ignore[invalid-argument-type]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(plutus_router)
app.include_router(chat_router)
app.include_router(models_router)
app.include_router(embeddings_router)
app.include_router(health_router)


def main() -> None:
    """Main entry point for the Ouranos ML API."""
    settings = get_settings()
    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
