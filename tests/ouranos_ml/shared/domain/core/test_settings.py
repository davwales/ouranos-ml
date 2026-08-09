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
