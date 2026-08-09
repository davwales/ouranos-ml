"""Pytest root-level configuration."""

import sys
from unittest.mock import MagicMock

try:
    import torch  # noqa: F401
except (ImportError, OSError):
    sys.modules["torch"] = MagicMock()
    sys.modules.setdefault("experiments", MagicMock())
    sys.modules.setdefault("experiments.plutus_forecasting", MagicMock())
    sys.modules.setdefault("experiments.plutus_forecasting.model", MagicMock())
    sys.modules.setdefault("experiments.utils", MagicMock())
    sys.modules.setdefault("experiments.utils.harness", MagicMock())
