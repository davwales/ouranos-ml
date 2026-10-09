from collections.abc import AsyncGenerator, AsyncIterator

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
from openai.types.responses.response_reasoning_item_param import Summary
from openai.types.shared_params import Reasoning as ReasoningParam
from openai.types.shared_params import ResponseFormatText as ResponseFormatTextParam

from ouranos_ml.features.responses.create_response.schemas import (
    STREAM_EVENT_ADAPTER,
    CreateResponseRequest,
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
        upstream = await get_openai_client().responses.create(**_to_create_params(request))
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
        stream = await get_openai_client().responses.create(**_to_create_params(request), stream=True)
    except Exception:
        logger.error("response upstream failed", model=request.model, exc_info=True)
        raise

    return _relay_stream(stream, request)


async def _relay_stream(
    stream: AsyncStream[SDKResponseStreamEvent], request: CreateResponseRequest
) -> AsyncGenerator[ResponseStreamEvent]:
    """Convert and relay upstream events, skipping event types we do not model."""
    try:
        async with stream:
            async for upstream_event in stream:
                try:
                    event = STREAM_EVENT_ADAPTER.validate_python(upstream_event.to_dict())
                except ValidationError as exc:
                    if exc.errors()[0]["type"] != "union_tag_invalid":
                        raise
                    logger.debug("unsupported stream event skipped", event_type=upstream_event.type)
                    continue
                if isinstance(event, ResponseSnapshotEvent) and event.type == "response.completed":
                    _log_complete(event.response)
                yield event
    except Exception:
        logger.error("response stream failed", model=request.model, exc_info=True)
        raise


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
            return ResponseReasoningItemParam(
                type="reasoning",
                id=item.id,
                summary=[Summary(type="summary_text", text=part.text) for part in item.summary],
                encrypted_content=item.encrypted_content,
            )


def _to_content_param(part: InputContent) -> ResponseInputContentParam:
    """Convert a content part; replayed ``output_text`` is sent as text, which is all the upstream reads."""
    if isinstance(part, InputImage):
        return ResponseInputImageParam(type="input_image", image_url=part.image_url, detail=part.detail)
    return ResponseInputTextParam(type="input_text", text=part.text)


def _to_tool_param(tool: FunctionTool) -> FunctionToolParam:
    return FunctionToolParam(
        type="function", name=tool.name, description=tool.description, parameters=tool.parameters, strict=tool.strict
    )


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
