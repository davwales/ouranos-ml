"""App-level fixtures for all ouranos_ml tests."""

from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Provide an httpx AsyncClient backed by the FastAPI ASGI app.

    Depends on a per-test-module ``app`` fixture (each ``test_endpoint.py``
    defines its own, mounting only the router under test for isolation).
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
