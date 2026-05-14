from functools import lru_cache

from openai import AsyncOpenAI

from ouranos_ml.shared.domain.core.settings import get_settings


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    """Creates the OpenAI-compatible client for the LLM service."""
    settings = get_settings()
    return AsyncOpenAI(api_key="ouranos_ml", base_url=settings.llm_openai_base_url)
