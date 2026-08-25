"""Pytest root-level configuration."""

import sys
from unittest.mock import MagicMock

try:
    import torch  # noqa: F401
except (ImportError, OSError):
    sys.modules["torch"] = MagicMock()
    sys.modules.setdefault("ouranos_ml.shared.inference", MagicMock())
    sys.modules.setdefault("ouranos_ml.shared.inference.model", MagicMock())
    sys.modules.setdefault("ouranos_ml.shared.inference.harness", MagicMock())