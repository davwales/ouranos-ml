"""Tests for ouranos_ml.shared.logging.config."""

import json
import logging

import pytest
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from ouranos_ml.shared.domain.core.settings import Settings
from ouranos_ml.shared.logging.config import ConsoleHandler, configure_logging
from ouranos_ml.shared.logging.loki import LokiPushHandler


def _settings(**overrides: object) -> Settings:
    """Build settings for logging configuration tests."""
    defaults: dict[str, object] = {"log_level": "INFO", "log_json": True}
    defaults.update(overrides)
    return Settings(**defaults)


def _own_handlers() -> list[logging.Handler]:
    """Return the handlers currently installed by this module on the root logger."""
    return [h for h in logging.getLogger().handlers if isinstance(h, (ConsoleHandler, LokiPushHandler))]


def test_configure_logging_when_no_loki_should_install_console_handler_only() -> None:
    """Test that a configuration without a Loki URL installs a single console handler."""
    # Act
    configure_logging(_settings())

    # Assert
    handlers = _own_handlers()
    assert len(handlers) == 1
    assert isinstance(handlers[0], ConsoleHandler)
    assert not any(isinstance(h, LokiPushHandler) for h in logging.getLogger().handlers)


def test_configure_logging_when_loki_enabled_should_add_loki_handler_and_restrict_console() -> None:
    """Test that a Loki base URL adds the push handler and drops the console to WARNING."""
    # Act
    configure_logging(_settings(loki_base_url="http://loki.test:3100"))

    # Assert
    handlers = _own_handlers()
    console = [h for h in handlers if isinstance(h, ConsoleHandler)]
    loki = [h for h in handlers if isinstance(h, LokiPushHandler)]
    assert len(console) == 1
    assert len(loki) == 1
    assert console[0].level == logging.WARNING


def test_configure_logging_when_log_json_false_should_render_pretty_lines(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that JSON disabled falls back to the pretty console renderer."""
    # Arrange
    configure_logging(_settings(log_json=False))

    # Act
    structlog.get_logger("test.native").info("pretty message", model="m")

    # Assert
    output = capsys.readouterr().out
    assert "pretty message" in output
    assert not output.lstrip().startswith("{")


def test_configure_logging_when_json_enabled_should_render_native_and_foreign_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that structlog and stdlib records share one JSON rendering pipeline."""
    # Arrange
    configure_logging(_settings())

    # Act
    structlog.get_logger("test.native").info("native event", model="llama")
    logging.getLogger("test.foreign").warning("foreign %s", "event")

    # Assert
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    native = next(line for line in lines if line["logger"] == "test.native")
    foreign = next(line for line in lines if line["logger"] == "test.foreign")
    assert native["message"] == "native event"
    assert native["model"] == "llama"
    assert native["level"] == "info"
    assert foreign["message"] == "foreign event"
    assert foreign["level"] == "warn"


def test_configure_logging_when_exception_record_should_render_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that exception info from stdlib records is rendered into the line."""
    # Arrange
    configure_logging(_settings())

    # Act
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("test.err").exception("request crashed")

    # Assert
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["message"] == "request crashed"
    assert "ValueError: boom" in line["exception"]


def test_configure_logging_when_request_id_bound_should_merge_into_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that contextvars bound around a request flow into rendered lines."""
    # Arrange
    configure_logging(_settings())
    clear_contextvars()
    bind_contextvars(request_id="req-123")

    # Act
    structlog.get_logger("test.ctx").info("inside request")

    # Assert
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["request_id"] == "req-123"


def test_configure_logging_when_root_level_info_should_filter_debug(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that events below the configured level are not rendered."""
    # Arrange
    configure_logging(_settings(log_level="INFO"))

    # Act
    structlog.get_logger("test.filter").debug("too chatty")
    structlog.get_logger("test.filter").info("just right")

    # Assert
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    messages = [line["message"] for line in lines]
    assert "too chatty" not in messages
    assert "just right" in messages


def test_configure_logging_when_levels_should_normalize_to_grafana_vocabulary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that warning and critical records render as warn and fatal."""
    # Arrange
    configure_logging(_settings())

    # Act
    structlog.get_logger("test.levels.native").warning("warned")
    logging.getLogger("test.levels.foreign").critical("critical event")

    # Assert
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    levels = {line["logger"].split(".")[-1]: line["level"] for line in lines}
    assert levels["native"] == "warn"
    assert levels["foreign"] == "fatal"


def test_configure_logging_when_invalid_level_should_raise_value_error() -> None:
    """Test that an unrecognized LOG_LEVEL fails fast."""
    # Act & Assert
    with pytest.raises(ValueError, match="Invalid LOG_LEVEL"):
        configure_logging(_settings(log_level="LOUD"))


def test_configure_logging_when_rerun_should_remove_only_own_handlers() -> None:
    """Test that reconfiguration keeps foreign handlers such as pytest's capture."""
    # Arrange
    configure_logging(_settings())
    foreign = logging.Handler()
    logging.getLogger().addHandler(foreign)

    # Act
    configure_logging(_settings())

    # Assert
    assert foreign in logging.getLogger().handlers
    assert len(_own_handlers()) == 1


def test_configure_logging_when_noisy_loggers_should_be_restricted_to_warning() -> None:
    """Test that third-party noisy loggers are dialed down to WARNING."""
    # Act
    configure_logging(_settings())

    # Assert
    for name in ("httpx", "openai", "matplotlib"):
        assert logging.getLogger(name).level == logging.WARNING
