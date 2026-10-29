import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import openai
import structlog
from openai import AsyncStream
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseFunctionToolCallParam,
    ResponseInputContentParam,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputTextParam,
    ResponseReasoningItemParam,
    ResponseTextConfigParam,
)
from openai.types.responses import ResponseStreamEvent as SDKResponseStreamEvent
from openai.types.responses.response_create_params import ResponseCreateParamsBase
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput as FunctionCallOutputParam
from openai.types.responses.response_reasoning_item_param import Content as ReasoningContentParam
from openai.types.responses.response_reasoning_item_param import Summary
from openai.types.shared_params import Reasoning as ReasoningParam
from openai.types.shared_params import ResponseFormatText as ResponseFormatTextParam

from ouranos_ml.features.responses.create_response.schemas import (
    STREAM_EVENT_ADAPTER,
    CreateResponseRequest,
    ErrorEvent,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    InputContent,
    InputImage,
    InputItem,
    InputMessage,
    Reasoning,
    ReasoningConfig,
    ResponseObject,
    ResponseSnapshotEvent,
    ResponseStreamEvent,
    TextConfig,
)
from ouranos_ml.shared.infra.clients.llm_client import get_openai_client
from pydantic import ValidationError

logger = structlog.get_logger(__name__)


async def handle(request: CreateResponseRequest) -> ResponseObject:
    """Create a model response and return it in full."""
    _log_requested(request)

    try:
        upstream = await get_openai_client().responses.create(
            **_to_create_params(request), extra_body=_to_extra_body(request)
        )
    except Exception:
        logger.error("response upstream failed", model=request.model, exc_info=True)
        raise

    response = ResponseObject.model_validate(upstream.to_dict())
    _log_complete(response)
    return response


async def open_stream(request: CreateResponseRequest) -> AsyncIterator[ResponseStreamEvent]:
    """Open the upstream stream eagerly, then return an iterator over its converted events.

    Opening eagerly means connection failures and upstream 4xx/5xx responses raise here, before
    any SSE bytes are sent, so they surface as real HTTP status codes.
    """
    _log_requested(request)
    try:
        stream = await get_openai_client().responses.create(
            **_to_create_params(request), extra_body=_to_extra_body(request), stream=True
        )
    except Exception:
        logger.error("response upstream failed", model=request.model, exc_info=True)
        raise

    return _relay_stream(stream, request)


async def _relay_stream(
    stream: AsyncStream[SDKResponseStreamEvent], request: CreateResponseRequest
) -> AsyncGenerator[ResponseStreamEvent]:
    """Convert and relay upstream events, ending with an ``error`` event instead of raising.

    Event types we do not model are skipped quietly; a malformed event of a modeled type is skipped with a warning.
    """
    backfill = _StreamBackfill(request.model)
    try:
        async with stream:
            async for upstream_event in stream:
                try:
                    event = STREAM_EVENT_ADAPTER.validate_python(backfill(upstream_event.to_dict()))
                except ValidationError as exc:
                    if exc.errors()[0]["type"] == "union_tag_invalid":
                        logger.debug("unsupported stream event skipped", event_type=upstream_event.type)
                    else:
                        logger.warning(
                            "malformed stream event skipped",
                            event_type=upstream_event.type,
                            error_count=exc.error_count(),
                        )
                    continue
                if isinstance(event, ResponseSnapshotEvent) and event.type == "response.completed":
                    _log_complete(event.response)
                yield event
    except Exception as exc:
        logger.error("response stream failed", model=request.model, exc_info=True)
        yield _error_event(exc, backfill.next_sequence_number())


def _error_event(exc: Exception, sequence_number: int) -> ErrorEvent:
    """Describe a mid-stream failure; codes follow the HTTP error bodies (``timeout`` here, as the 504's)."""
    if isinstance(exc, openai.APITimeoutError):
        return ErrorEvent(type="error", sequence_number=sequence_number, code="timeout", message=exc.message)
    message = (
        exc.message if isinstance(exc, openai.APIError) else f"The LLM backend stream failed ({type(exc).__name__})"
    )
    return ErrorEvent(type="error", sequence_number=sequence_number, code="server_error", message=message)


class _StreamBackfill:
    """Fill in stream event fields the upstream omits, keeping every field it does send.

    llama.cpp's ``llama-server`` leaves out ``sequence_number`` on every event, ``output_index``/``content_index``
    on item, part and delta events, and ``created_at``/``model``/``output`` on the ``response.created`` and
    ``response.in_progress`` snapshots. Clients deserialize these as required fields.
    """

    def __init__(self, model: str) -> None:
        self._model = model
        self._created_at = int(time.time())
        self._sequence_number = 0
        self._item_ids: list[str | None] = []

    def __call__(self, event: dict[str, Any]) -> dict[str, Any]:
        sequence_number = event.setdefault("sequence_number", self._sequence_number)
        self._sequence_number = (sequence_number if isinstance(sequence_number, int) else self._sequence_number) + 1
        if isinstance(item := event.get("item"), dict):
            event.setdefault("output_index", self._output_index(item.get("id"), add=True))
        elif "item_id" in event:
            event.setdefault("output_index", self._output_index(event["item_id"], add=False))
            event.setdefault("content_index", 0)
        if isinstance(response := event.get("response"), dict):
            response.setdefault("created_at", self._created_at)
            response.setdefault("model", self._model)
            response.setdefault("output", [])
        return event

    def next_sequence_number(self) -> int:
        """The sequence number for an event we add after the last upstream one."""
        sequence_number = self._sequence_number
        self._sequence_number += 1
        return sequence_number

    def _output_index(self, item_id: str | None, *, add: bool) -> int:
        """Position of the item in the output, in the order the upstream added items."""
        if item_id is not None and item_id in self._item_ids:
            return self._item_ids.index(item_id)
        if not add:
            return 0
        self._item_ids.append(item_id)
        return len(self._item_ids) - 1


def _to_create_params(request: CreateResponseRequest) -> ResponseCreateParamsBase:
    """Build the upstream request; unset optional fields are left out rather than sent as ``null``."""
    params = ResponseCreateParamsBase(model=request.model, input=_to_sdk_input(request.input))
    if request.instructions is not None:
        params["instructions"] = request.instructions
    if request.tools:
        params["tools"] = [_to_tool_param(tool) for tool in request.tools]
    if request.text is not None:
        params["text"] = _to_text_param(request.text)
    if request.reasoning is not None:
        params["reasoning"] = _to_reasoning_param(request.reasoning)
    if request.temperature is not None:
        params["temperature"] = request.temperature
    if request.top_p is not None:
        params["top_p"] = request.top_p
    if request.max_output_tokens is not None:
        params["max_output_tokens"] = request.max_output_tokens
    return params


def _to_extra_body(request: CreateResponseRequest) -> dict[str, Any] | None:
    """Mirror a JSON schema text format as a top-level ``response_format``, sent alongside ``text``.

    llama.cpp's ``llama-server`` ignores ``text.format`` on ``/responses`` but constrains the output to a
    ``response_format`` (it reads only ``json_schema.schema``); backends that honour ``text.format`` ignore the extra.
    """
    if request.text is None or request.text.format.type != "json_schema":
        return None
    json_schema = {"name": request.text.format.name, "schema": request.text.format.json_schema}
    return {"response_format": {"type": "json_schema", "json_schema": json_schema}}


def _to_sdk_input(input_value: str | list[InputItem]) -> str | list[ResponseInputItemParam]:
    if isinstance(input_value, str):
        return input_value
    return [_to_item_param(item) for item in input_value]


def _to_item_param(item: InputItem) -> ResponseInputItemParam:
    match item:
        case InputMessage():
            content = item.content if isinstance(item.content, str) else [_to_content_param(p) for p in item.content]
            return EasyInputMessageParam(type="message", role=item.role, content=content)
        case FunctionCall():
            param = ResponseFunctionToolCallParam(
                type="function_call", call_id=item.call_id, name=item.name, arguments=item.arguments
            )
            if item.id is not None:
                param["id"] = item.id
            return param
        case FunctionCallOutput():
            return FunctionCallOutputParam(type="function_call_output", call_id=item.call_id, output=item.output)
        case Reasoning():
            param = ResponseReasoningItemParam(
                type="reasoning",
                id=item.id,
                summary=[Summary(type="summary_text", text=part.text) for part in item.summary],
                encrypted_content=item.encrypted_content,
            )
            if item.content:
                param["content"] = [
                    ReasoningContentParam(type="reasoning_text", text=part.text) for part in item.content
                ]
            return param


def _to_content_param(part: InputContent) -> ResponseInputContentParam:
    """Convert a content part; replayed ``output_text`` is sent as text, which is all the upstream reads."""
    if isinstance(part, InputImage):
        return ResponseInputImageParam(type="input_image", image_url=part.image_url, detail=part.detail)
    return ResponseInputTextParam(type="input_text", text=part.text)


def _to_tool_param(tool: FunctionTool) -> FunctionToolParam:
    """Convert a function tool; an unset description is left out (llama.cpp rejects a ``null`` one)."""
    param = FunctionToolParam(type="function", name=tool.name, parameters=tool.parameters, strict=tool.strict)
    if tool.description is not None:
        param["description"] = tool.description
    return param


def _to_text_param(text: TextConfig) -> ResponseTextConfigParam:
    if text.format.type == "text":
        return ResponseTextConfigParam(format=ResponseFormatTextParam(type="text"))
    json_format = ResponseFormatTextJSONSchemaConfigParam(
        type="json_schema", name=text.format.name, schema=text.format.json_schema, strict=text.format.strict
    )
    if text.format.description is not None:
        json_format["description"] = text.format.description
    return ResponseTextConfigParam(format=json_format)


def _to_reasoning_param(reasoning: ReasoningConfig) -> ReasoningParam:
    return ReasoningParam(effort=reasoning.effort, summary=reasoning.summary)


def _log_requested(request: CreateResponseRequest) -> None:
    logger.debug(
        "response requested",
        model=request.model,
        input_item_count=len(request.input) if isinstance(request.input, list) else 1,
        tool_count=len(request.tools or []),
        stream=request.stream,
    )


def _log_complete(response: ResponseObject) -> None:
    logger.debug(
        "response complete",
        model=response.model,
        status=response.status,
        input_tokens=response.usage.input_tokens if response.usage else None,
        output_tokens=response.usage.output_tokens if response.usage else None,
    )
