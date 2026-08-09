"""Shared mock helpers for OpenAI streaming chat completions.

These helpers fake the async context manager + async iterator protocol used by
``client.chat.completions.stream(...)`` so service- and endpoint-layer tests can
drive the streaming code path without a live LLM backend.
"""

from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import AsyncMock, MagicMock

from openai.lib.streaming.chat import ChunkEvent


class AsyncIteratorMock:
    """Mock async iterator that yields items from a synchronous iterable."""

    def __init__(self, items: Iterable[ChunkEvent]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> AsyncIteratorMock:
        return self

    async def __anext__(self) -> ChunkEvent:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def make_mock_stream(events: Iterable[ChunkEvent]) -> MagicMock:
    """Build a mock async context manager that yields ``events`` as an async iterable.

    The returned mock satisfies ``async with client.chat.completions.stream(...) as stream:``
    followed by ``async for event in stream:``.
    """
    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)
    mock_stream.__aiter__ = MagicMock(return_value=AsyncIteratorMock(events))
    return mock_stream
