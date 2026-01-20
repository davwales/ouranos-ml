import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ouranos_ml.features.chat.router import chat_router
from ouranos_ml.features.generation.router import generation_router
from ouranos_ml.features.plutus.router import plutus_router

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
app.include_router(generation_router)  # remove once chat route is fully deprecated


def main() -> None:
    """Main entry point for the Ouranos ML API."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
