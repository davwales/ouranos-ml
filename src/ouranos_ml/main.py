import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ouranos_ml.features.chat.router import chat_router
from ouranos_ml.features.embeddings.router import embeddings_router
from ouranos_ml.features.health.router import health_router
from ouranos_ml.features.images.router import images_router
from ouranos_ml.features.models.router import models_router
from ouranos_ml.features.plutus.router import plutus_router
from ouranos_ml.features.responses.router import responses_router
from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.infra.openai.errors import register_openai_error_handler
from ouranos_ml.shared.infra.swarmui.errors import register_swarmui_error_handler
from ouranos_ml.shared.logging import RequestLoggingMiddleware, configure_logging

app = FastAPI()

# Added last, so CORS wraps request logging and its JSON 500s get CORS headers too.
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
register_openai_error_handler(app)
register_swarmui_error_handler(app)

app.include_router(chat_router)
app.include_router(embeddings_router)
app.include_router(health_router)
app.include_router(images_router)
app.include_router(models_router)
app.include_router(plutus_router)
app.include_router(responses_router)


def main() -> None:
    """Main entry point for the Ouranos ML API."""
    settings = get_settings()
    configure_logging(settings)
    uvicorn.run(app, host="0.0.0.0", port=settings.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
