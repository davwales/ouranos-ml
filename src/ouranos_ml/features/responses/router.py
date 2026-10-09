from fastapi import APIRouter

from ouranos_ml.features.responses.create_response.endpoint import register as register_create_response

responses_router = APIRouter(prefix="/responses")
register_create_response(responses_router)
