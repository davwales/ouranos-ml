from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurable values for the Ouranos ML application."""

    lmstudio_base_url: str = "localhost:1234"
    lmstudio_openai_base_url: str = "http://localhost:1234/v1"
    lmstudio_model_ttl: int = 300

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Gets the settings instance."""
    return Settings()
