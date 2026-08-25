from enum import StrEnum
from typing import Annotated, Any, Literal

from ouranos_ml.shared.domain.core.base_schema import BaseSchema
from pydantic import AliasChoices, Field


class ChatCompletionRole(StrEnum):
    """Represents what role an actor in a chat is."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class RequestMessage(BaseSchema):
    """Represents a single request message in the chat."""

    role: ChatCompletionRole
    content: str


class StreamOptions(BaseSchema):
    """Options for streaming response. Only set when stream is true."""

    include_usage: bool = False
    """If set, an additional chunk will be streamed before the [DONE] message
    with total token usage. All other chunks will include a null usage field."""


class JSONSchemaConfig(BaseSchema):
    """Structured Outputs configuration, including the JSON Schema to conform to."""

    name: str
    """The name of the response format. Must be a-z, A-Z, 0-9, or contain
    underscores and dashes, with a maximum length of 64."""

    description: str | None = None
    """A description of what the response format is for, used by the model to
    determine how to respond in the format."""

    strict: bool | None = None
    """Whether to enable strict schema adherence when generating the output.
    If set to true, the model will always follow the exact schema defined in
    the schema field."""

    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    """The schema for the response format, described as a JSON Schema object.

    Uses the OpenAI wire name ``schema``; the Python field name avoids shadowing
    the deprecated ``BaseModel.schema`` classmethod.
    """


class ResponseFormatText(BaseSchema):
    """Default response format used to generate text responses."""

    type: Literal["text"] = "text"


class ResponseFormatJSONObject(BaseSchema):
    """JSON object response format, forcing the model to output valid JSON."""

    type: Literal["json_object"]


class ResponseFormatJSONSchema(BaseSchema):
    """JSON Schema response format used to generate structured JSON responses."""

    type: Literal["json_schema"]
    json_schema: JSONSchemaConfig = Field(
        alias="json_schema", validation_alias=AliasChoices("jsonSchema", "json_schema")
    )


ResponseFormat = Annotated[
    ResponseFormatText | ResponseFormatJSONObject | ResponseFormatJSONSchema,
    Field(discriminator="type"),
]


class ChatCompletionsRequest(BaseSchema):
    """Request for generating a chat completion."""

    model: str
    messages: list[RequestMessage]

    top_p: float = 1.0
    temperature: float = 1.0
    max_completion_tokens: int | None = None
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    response_format: ResponseFormat | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, int] | None = None


class ChoiceMessage(BaseSchema):
    """Message in a chat completion response."""

    role: ChatCompletionRole
    content: str
    refusal: str | None = None


class Choice(BaseSchema):
    """Choice from the chat completion."""

    finish_reason: str
    index: int
    message: ChoiceMessage


class Usage(BaseSchema):
    """Usage statistics for the chat completion"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionsResponse(BaseSchema):
    """Response for chat completions."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    stats: dict[str, Any] = {}
    system_fingerprint: str | None


class ChunkDelta(BaseSchema):
    """Delta for a chunk."""

    content: str | None


class ChunkChoice(BaseSchema):
    """Choice from a streamed chat."""

    index: int
    delta: ChunkDelta
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None


class ChatCompletionChunkResponse(BaseSchema):
    """Response for a streamed chat completion."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    system_fingerprint: str | None
    choices: list[ChunkChoice]
    usage: Usage | None = None
