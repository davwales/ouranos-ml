from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurable values for the Ouranos ML application."""

    llm_openai_base_url: str = "http://localhost:11434/v1"
    health_check_timeout_seconds: float = 5.0
    port: int = 8000
    models_dir: str = "models"
    plutus_forecast_model_name: str = "plutus-forecasting-v1"
    log_level: str = "INFO"
    log_json: bool | None = None
    log_app_name: str = "Ouranos.Ml"
    loki_base_url: str = ""
    loki_tenant_id: str = "tenant1"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Gets the settings instance."""
    return Settings()
