import httpx
import pytest
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


def test_get_openai_client_when_called_should_use_default_timeout_and_no_retries() -> None:
    """Test that get_openai_client applies the default timeout and disables retries."""
    # Arrange
    get_openai_client.cache_clear()
    get_settings.cache_clear()

    # Act
    client = get_openai_client()

    # Assert
    assert client.timeout == httpx.Timeout(300.0, connect=5.0)
    assert client.max_retries == 0


def test_get_openai_client_when_env_vars_set_should_use_configured_timeout_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that get_openai_client passes the configured timeout and max retries to the SDK."""
    # Arrange
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    get_openai_client.cache_clear()
    get_settings.cache_clear()

    # Act
    client = get_openai_client()

    # Assert
    assert client.timeout == httpx.Timeout(12.5, connect=5.0)
    assert client.max_retries == 3
