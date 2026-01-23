from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from lmstudio import AsyncClient

from ouranos_ml.shared.domain.core.settings import get_settings


@asynccontextmanager
async def get_client() -> AsyncGenerator[AsyncClient]:
    """Creates an async LMStudio client."""
    settings = get_settings()
    async with AsyncClient(settings.lmstudio_base_url) as client:
        yield client
