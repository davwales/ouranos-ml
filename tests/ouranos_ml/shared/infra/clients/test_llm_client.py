from openai import AsyncOpenAI

from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.infra.clients.llm_client import get_openai_client


def test_get_openai_client_when_called_should_return_async_openai() -> None:
    """Test that get_openai_client returns an AsyncOpenAI instance."""
    # Arrange
    get_openai_client.cache_clear()
    get_settings.cache_clear()

    # Act
    client = get_openai_client()

    # Assert
    assert isinstance(client, AsyncOpenAI)


def test_get_openai_client_when_called_should_use_configured_base_url() -> None:
    """Test that get_openai_client uses the base URL from settings."""
    # Arrange
    get_openai_client.cache_clear()
    get_settings.cache_clear()
    settings = get_settings()

    # Act
    client = get_openai_client()

    # Assert
    assert str(client.base_url).rstrip("/") == settings.llm_openai_base_url.rstrip("/")


def test_get_openai_client_when_called_twice_should_return_same_instance() -> None:
    """Test that get_openai_client returns the cached instance on repeated calls."""
    # Arrange
    get_openai_client.cache_clear()
    get_settings.cache_clear()

    # Act
    first = get_openai_client()
    second = get_openai_client()

    # Assert
    assert first is second
