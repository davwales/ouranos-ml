"""Factory functions for creating Responses API test data.

Upstream objects are built as real ``openai`` SDK types (validated from wire-shaped dicts) so the
service's conversion into our own schemas is exercised exactly as it is against a live backend.
"""

from typing import Any

from openai.types.responses import Response as SDKResponse
from openai.types.responses import ResponseStreamEvent

from ouranos_ml.features.responses.create_response.schemas import CreateResponseRequest
from pydantic import TypeAdapter

_STREAM_EVENT_ADAPTER: TypeAdapter[ResponseStreamEvent] = TypeAdapter(ResponseStreamEvent)


def make_function_tool(*, name: str = "get_weather") -> dict[str, Any]:
    """Create a wire-format function tool definition."""
    return {
        "type": "function",
        "name": name,
        "description": "Get the weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        "strict": True,
    }


def make_create_response_request(**overrides: Any) -> CreateResponseRequest:
    """Create a CreateResponseRequest from wire-format fields with sensible defaults."""
    payload: dict[str, Any] = {"model": "test-model", "input": "Hello, world"}
    payload.update(overrides)
    return CreateResponseRequest.model_validate(payload)


def make_output_message(*, item_id: str = "msg_001", text: str = "Hello, world", status: str = "completed") -> dict:
    """Create a wire-format assistant output message item."""
    content = [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}] if text else []
    return {"type": "message", "id": item_id, "role": "assistant", "status": status, "content": content}


def make_function_call_item(*, item_id: str = "fc_001", arguments: str = '{"city":"Paris"}') -> dict:
    """Create a wire-format function call output item."""
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": "call_001",
        "name": "get_weather",
        "arguments": arguments,
        "status": "completed",
    }


def make_response_payload(
    *,
    status: str = "completed",
    output: list[dict] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    with_usage: bool = True,
    text: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a wire-format response object shaped like the upstream's."""
    return {
        "id": "resp_001",
        "object": "response",
        "created_at": 1700000000,
        "completed_at": 1700000001 if status == "completed" else None,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "model": "test-model",
        "output": [make_output_message()] if output is None else output,
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}} if text is None else text,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "metadata": {},
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        }
        if with_usage
        else None,
    }


def make_sdk_response(**kwargs: Any) -> SDKResponse:
    """Create an SDK ``Response`` as returned by ``client.responses.create``."""
    return SDKResponse.model_validate(make_response_payload(**kwargs))


def make_stream_event(event_type: str, sequence_number: int, **fields: Any) -> ResponseStreamEvent:
    """Create an SDK stream event of any type from its wire fields."""
    return _STREAM_EVENT_ADAPTER.validate_python({"type": event_type, "sequence_number": sequence_number, **fields})


def make_text_stream_events(
    *, text_deltas: tuple[str, ...] = ("Hello", ", world"), item_id: str = "msg_001"
) -> list[ResponseStreamEvent]:
    """Create a spec-ordered SDK event sequence for a streamed text response."""
    text = "".join(text_deltas)
    in_progress = make_response_payload(status="in_progress", output=[], with_usage=False)
    part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
    location = {"item_id": item_id, "output_index": 0, "content_index": 0}
    payloads: list[tuple[str, dict[str, Any]]] = [
        ("response.created", {"response": in_progress}),
        ("response.in_progress", {"response": in_progress}),
        ("response.output_item.added", {"output_index": 0, "item": make_output_message(item_id=item_id, text="")}),
        ("response.content_part.added", {**location, "part": part}),
        *[("response.output_text.delta", {**location, "delta": delta, "logprobs": []}) for delta in text_deltas],
        ("response.output_text.done", {**location, "text": text, "logprobs": []}),
        ("response.content_part.done", {**location, "part": {**part, "text": text}}),
        ("response.output_item.done", {"output_index": 0, "item": make_output_message(item_id=item_id, text=text)}),
        (
            "response.completed",
            {"response": make_response_payload(output=[make_output_message(item_id=item_id, text=text)])},
        ),
    ]
    return [make_stream_event(event_type, index, **fields) for index, (event_type, fields) in enumerate(payloads)]
