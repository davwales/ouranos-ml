"""Shared fixtures for the logging package tests."""

import logging
from collections.abc import Iterator

import pytest
import structlog

from ouranos_ml.shared.logging.loki import LokiPushHandler


@pytest.fixture(autouse=True)
def isolate_logging() -> Iterator[None]:
    """Snapshot and restore global logging and structlog state around each test."""
    saved_config = structlog.get_config()
    saved_handlers = logging.root.handlers[:]
    saved_level = logging.root.level

    yield

    # Restore: stop handler threads before restoring state
    for handler in logging.root.handlers:
        if isinstance(handler, LokiPushHandler):
            handler.close()
    structlog.configure(**saved_config)
    logging.root.handlers[:] = saved_handlers
    logging.root.setLevel(saved_level)
