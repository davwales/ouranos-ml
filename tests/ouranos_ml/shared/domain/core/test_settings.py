import pytest

from ouranos_ml.shared.domain.core.settings import Settings, get_settings


def test_get_settings_when_called_should_return_settings_instance() -> None:
    """Test that get_settings returns a Settings instance."""
    # Arrange
    get_settings.cache_clear()

    # Act
    result = get_settings()

    # Assert
    assert isinstance(result, Settings)


def test_settings_when_defaults_should_have_expected_values() -> None:
    """Test that Settings class defaults match expected values."""
    # Arrange
    get_settings.cache_clear()

    # Act
    defaults = Settings.model_fields

    # Assert
    assert defaults["port"].default == 8000
    assert defaults["llm_openai_base_url"].default == "http://localhost:11434/v1"
    assert defaults["llm_base_url"].default == "localhost:11434"
    assert defaults["llm_model_ttl"].default == 300
    assert defaults["health_check_timeout_seconds"].default == 5.0
    assert defaults["log_level"].default == "INFO"
    assert defaults["log_json"].default is None
    assert defaults["log_app_name"].default == "Ouranos.Ml"
    assert defaults["loki_base_url"].default == ""
    assert defaults["loki_tenant_id"].default == "tenant1"


def test_get_settings_when_called_twice_should_return_same_instance() -> None:
    """Test that get_settings returns the cached instance on repeated calls."""
    # Arrange
    get_settings.cache_clear()

    # Act
    first = get_settings()
    second = get_settings()

    # Assert
    assert first is second


def test_get_settings_when_cache_cleared_should_return_fresh_instance() -> None:
    """Test that clearing the cache causes get_settings to return a new instance."""
    # Arrange
    get_settings.cache_clear()
    first = get_settings()

    # Act
    get_settings.cache_clear()
    second = get_settings()

    # Assert
    assert first is not second


def test_settings_when_loki_env_vars_set_should_parse_logging_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logging fields bind from environment variables."""
    # Arrange
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_JSON", "true")
    monkeypatch.setenv("LOG_APP_NAME", "Test.App")
    monkeypatch.setenv("LOKI_BASE_URL", "http://loki:3100")
    monkeypatch.setenv("LOKI_TENANT_ID", "acme")

    # Act
    settings = Settings()

    # Assert
    assert settings.log_level == "DEBUG"
    assert settings.log_json is True
    assert settings.log_app_name == "Test.App"
    assert settings.loki_base_url == "http://loki:3100"
    assert settings.loki_tenant_id == "acme"
