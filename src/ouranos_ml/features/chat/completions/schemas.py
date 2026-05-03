from enum import StrEnum
from typing import Any, Literal

from ouranos_ml.shared.domain.core.base_schema import BaseSchema


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


class ChatCompletionsRequest(BaseSchema):
    """Request for generating a chat completion."""

    model: str
    messages: list[RequestMessage]

    top_p: float = 1.0
    temperature: float = 1.0
    max_completion_tokens: int | None = None
    stream: bool | None = False
    stream_options: StreamOptions | None = None
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
