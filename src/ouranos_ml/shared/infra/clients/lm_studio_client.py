from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache

from lmstudio import AsyncClient
from openai import AsyncOpenAI

from ouranos_ml.shared.domain.core.settings import get_settings


@asynccontextmanager
async def get_client() -> AsyncGenerator[AsyncClient]:
    """Creates an async LMStudio client."""
    settings = get_settings()
    async with AsyncClient(settings.lmstudio_base_url) as client:
        yield client


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    """Creates the OpenAI compatible client with LMStudio."""
    settings = get_settings()
    return AsyncOpenAI(api_key="lmstudio", base_url=settings.lmstudio_openai_base_url)
