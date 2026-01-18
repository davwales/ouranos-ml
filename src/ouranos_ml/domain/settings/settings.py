from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurable values for the Ouranos ML application."""

    lmstudio_base_url: str = "localhost:1234"
    lmstudio_model_ttl: int = 300

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
