"""Pytest root-level configuration."""

import sys
from unittest.mock import MagicMock

import pytest

from ouranos_ml.shared.domain.core.settings import Settings

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    sys.modules["torch"] = MagicMock()
    sys.modules.setdefault("ouranos_ml.shared.inference", MagicMock())
    sys.modules.setdefault("ouranos_ml.shared.inference.model", MagicMock())
    sys.modules.setdefault("ouranos_ml.shared.inference.harness", MagicMock())


@pytest.fixture(autouse=True)
def hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate Settings from the developer's environment.

    Removes the .env file and any ambient environment variables that map to
    Settings fields, so every test sees the documented defaults unless it pins
    a value explicitly. Without this, any machine whose .env or shell sets
    LOKI_BASE_URL would see logging tests build a real Loki handler.
    """
    monkeypatch.setattr(Settings, "model_config", {**Settings.model_config, "env_file": None})
    for name in (
        "LLM_OPENAI_BASE_URL",
        "HEALTH_CHECK_TIMEOUT_SECONDS",
        "PORT",
        "MODELS_DIR",
        "PLUTUS_FORECAST_MODEL_NAME",
        "LOG_LEVEL",
        "LOG_JSON",
        "LOG_APP_NAME",
        "LOKI_BASE_URL",
        "LOKI_TENANT_ID",
    ):
        monkeypatch.delenv(name, raising=False)