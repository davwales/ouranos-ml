import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ouranos_ml.api.routes.generation import router as generation_router
from ouranos_ml.api.routes.plutus import router as plutus_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,  # ty: ignore[invalid-argument-type]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generation_router)
app.include_router(plutus_router)


def main() -> None:
    """Main entry point for the Ouranos ML API."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
