from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurable values for the Ouranos ML application."""

    llm_base_url: str = "localhost:11434"
    llm_openai_base_url: str = "http://localhost:11434/v1"
    llm_model_ttl: int = 300
    health_check_timeout_seconds: float = 5.0
    port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Gets the settings instance."""
    return Settings()
