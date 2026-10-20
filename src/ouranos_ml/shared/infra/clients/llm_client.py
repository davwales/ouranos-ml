from functools import lru_cache

import httpx
from openai import AsyncOpenAI

from ouranos_ml.shared.domain.core.settings import get_settings

_CONNECT_TIMEOUT_SECONDS = 5.0


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    """Creates the OpenAI-compatible client for the LLM service."""
    settings = get_settings()
    return AsyncOpenAI(
        api_key="ouranos_ml",
        base_url=settings.llm_openai_base_url,
        # Non-streaming backends send nothing until generation ends, so the read timeout bounds the whole call;
        # connect stays short so an unreachable backend fails fast.
        timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=_CONNECT_TIMEOUT_SECONDS),
        max_retries=settings.llm_max_retries,
    )
