from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from ouranos_ml.features.responses.create_response.schemas import CreateResponseRequest, ResponseObject
from ouranos_ml.features.responses.create_response.service import handle, open_stream
from ouranos_ml.shared.utils import server_side_event


async def _create_response(request: CreateResponseRequest) -> ResponseObject | StreamingResponse | JSONResponse:
    """Creates a model response, streamed as named SSE events (no ``[DONE]``) when ``stream`` is true.

    ``previous_response_id`` is rejected with an OpenAI-style 400: no responses are stored, so ignoring it
    would silently drop the conversation context the client expects.
    """
    if request.previous_response_id is not None:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "previous_response_id is not supported: this API is stateless and stores no "
                    "responses. Send the full conversation in input instead.",
                    "type": "invalid_request_error",
                    "param": "previous_response_id",
                },
            },
        )
    if request.stream:
        events = await open_stream(request)

        async def stream_events() -> AsyncIterator[str]:
            async for event in events:
                yield server_side_event(event, event=event.type)

        return StreamingResponse(stream_events(), media_type="text/event-stream")
    return await handle(request)


def register(router: APIRouter) -> None:
    """Register create-response endpoints on the provided router."""
    router.post(
        "",
        response_model=None,
        responses={
            200: {
                "model": ResponseObject,
                "description": "A response object as JSON, or a stream of named SSE events (no `[DONE]`) when `stream` is true.",
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
            }
        },
    )(_create_response)
