"""Tests for ouranos_ml.shared.logging.loki."""

import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from ouranos_ml.shared.logging.loki import LokiPushHandler, normalize_level_name, to_grafana_level

_BASE_URL = "http://loki.test:3100"
_TENANT = "tenant1"
_APP = "Ouranos.Ml"


class _JsonFormatter(logging.Formatter):
    """Minimal formatter producing parseable JSON lines for payload assertions."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"message": record.getMessage(), "level": record.levelname.lower()})


def _record(
    message: str,
    level: int = logging.INFO,
    name: str = "test.logger",
    created: float | None = None,
) -> logging.LogRecord:
    """Build a log record for handler tests."""
    record = logging.LogRecord(name, level, "test.py", 1, message, None, None)
    if created is not None:
        record.created = created
    return record


def _handler(client: MagicMock, **overrides: float) -> LokiPushHandler:
    """Build a LokiPushHandler with a mocked HTTP client and a long flush interval."""
    flush_interval = overrides.pop("flush_interval", 3600.0)
    handler = LokiPushHandler(
        base_url=_BASE_URL,
        tenant_id=_TENANT,
        app_name=_APP,
        batch_size=overrides.pop("batch_size", 100),
        flush_interval=flush_interval,
        client=client,
    )
    handler.setFormatter(_JsonFormatter())
    return handler


def _ok_response() -> SimpleNamespace:
    return SimpleNamespace(is_success=True)


def test_flush_when_records_buffered_should_post_batched_payload() -> None:
    """Test that flush groups records by level and posts a well-formed Loki payload."""
    # Arrange
    client = MagicMock(spec=httpx.Client)
    client.post.return_value = _ok_response()
    handler = _handler(client)
    handler.emit(_record("first", logging.INFO))
    handler.emit(_record("second", logging.DEBUG))
    handler.emit(_record("third", logging.INFO))

    # Act
    handler.flush()

    # Assert
    client.post.assert_called_once()
    assert client.post.call_args.args[0] == f"{_BASE_URL}/loki/api/v1/push"
    headers = client.post.call_args.kwargs["headers"]
    assert headers["X-Scope-OrgID"] == _TENANT
    assert headers["Content-Type"] == "application/json"

    payload = json.loads(client.post.call_args.kwargs["content"])
    streams = {stream["stream"]["level"]: stream for stream in payload["streams"]}
    assert set(streams) == {"info", "debug"}
    assert streams["info"]["stream"]["app"] == _APP
    assert [value[1] for value in streams["info"]["values"]] == [
        json.dumps({"message": "first", "level": "info"}),
        json.dumps({"message": "third", "level": "info"}),
    ]
    assert [value[1] for value in streams["debug"]["values"]] == [
        json.dumps({"message": "second", "level": "debug"}),
    ]

    handler.close()


def test_flush_when_timestamps_equal_should_keep_strictly_increasing_order() -> None:
    """Test that equal nanosecond timestamps within a stream are bumped apart."""
    # Arrange
    client = MagicMock(spec=httpx.Client)
    client.post.return_value = _ok_response()
    handler = _handler(client)
    handler.emit(_record("one", created=100.0))
    handler.emit(_record("two", created=100.0))

    # Act
    handler.flush()

    # Assert
    payload = json.loads(client.post.call_args.kwargs["content"])
    values = payload["streams"][0]["values"]
    assert int(values[0][0]) < int(values[1][0])
    assert int(values[1][0]) == int(values[0][0]) + 1

    handler.close()


def test_flush_when_transport_error_should_retry_once_then_drop() -> None:
    """Test that a failed push retries once, drops the batch, and never raises."""
    # Arrange
    client = MagicMock(spec=httpx.Client)
    client.post.side_effect = httpx.ConnectError("loki unreachable")
    handler = _handler(client)
    handler.emit(_record("lost"))

    # Act
    handler.flush()

    # Assert
    assert client.post.call_count == 2
    client.post.reset_mock()
    handler.flush()
    client.post.assert_not_called()

    handler.close()


def test_flush_when_error_response_should_retry_once_then_drop() -> None:
    """Test that non-2xx responses trigger one retry and then drop the batch."""
    # Arrange
    client = MagicMock(spec=httpx.Client)
    client.post.return_value = SimpleNamespace(is_success=False)
    handler = _handler(client)
    handler.emit(_record("lost"))

    # Act
    handler.flush()

    # Assert
    assert client.post.call_count == 2

    handler.close()


def test_flush_when_retry_succeeds_should_keep_batch_delivered() -> None:
    """Test that a batch is delivered when only the first attempt fails."""
    # Arrange
    client = MagicMock(spec=httpx.Client)
    client.post.side_effect = [httpx.ConnectError("blip"), _ok_response()]
    handler = _handler(client)
    handler.emit(_record("kept"))

    # Act
    handler.flush()

    # Assert
    assert client.post.call_count == 2

    handler.close()


def test_emit_when_format_fails_should_not_raise() -> None:
    """Test that a record which cannot be formatted never propagates an exception."""
    # Arrange
    client = MagicMock(spec=httpx.Client)
    handler = _handler(client)
    broken = logging.LogRecord("test.logger", logging.INFO, "test.py", 1, "%d %s", ("unmatched",), None)

    # Act & Assert
    handler.emit(broken)

    handler.close()


def test_close_when_pending_records_should_flush_and_be_idempotent() -> None:
    """Test that close pushes the remaining buffer and repeated calls are no-ops."""
    # Arrange
    client = MagicMock(spec=httpx.Client)
    client.post.return_value = _ok_response()
    handler = _handler(client)
    handler.emit(_record("tail"))

    # Act
    handler.close()
    handler.close()

    # Assert
    client.post.assert_called_once()


def test_flush_thread_when_interval_elapses_should_push_buffered_records() -> None:
    """Test that the background thread flushes after the flush interval."""
    # Arrange
    client = MagicMock(spec=httpx.Client)
    client.post.return_value = _ok_response()
    handler = _handler(client, flush_interval=0.05)

    try:
        handler.emit(_record("async"))

        # Act & Assert
        deadline = time.monotonic() + 2.0
        while client.post.call_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.post.call_count >= 1
    finally:
        handler.close()


def test_to_grafana_level_when_stdlib_levels_should_map_to_grafana_vocabulary() -> None:
    """Test that stdlib level numbers map onto the Grafana vocabulary."""
    # Act & Assert
    assert to_grafana_level(logging.DEBUG) == "debug"
    assert to_grafana_level(logging.INFO) == "info"
    assert to_grafana_level(logging.WARNING) == "warn"
    assert to_grafana_level(logging.ERROR) == "error"
    assert to_grafana_level(logging.CRITICAL) == "fatal"
    assert to_grafana_level(logging.WARNING - 1) == "info"


def test_normalize_level_name_when_aliases_should_map_to_grafana_vocabulary() -> None:
    """Test that stdlib level names normalize onto the Grafana vocabulary."""
    # Act & Assert
    assert normalize_level_name("WARNING") == "warn"
    assert normalize_level_name("CRITICAL") == "fatal"
    assert normalize_level_name("info") == "info"
    assert normalize_level_name("error") == "error"


def test_to_grafana_level_when_below_debug_should_return_debug() -> None:
    """Test that level numbers below DEBUG fall back to debug."""
    # Act & Assert
    assert to_grafana_level(logging.NOTSET) == "debug"


def test_emit_when_batch_size_reached_should_wake_flush_thread() -> None:
    """Test that filling the batch wakes the background thread immediately."""
    # Arrange
    client = MagicMock(spec=httpx.Client)
    client.post.return_value = _ok_response()
    handler = _handler(client, batch_size=2)

    try:
        handler.emit(_record("one"))
        handler.emit(_record("two"))

        # Act & Assert
        deadline = time.monotonic() + 2.0
        while client.post.call_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.post.call_count >= 1
    finally:
        handler.close()


def test_flush_when_push_raises_unexpectedly_should_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that an unexpected push failure is swallowed by flush."""
    # Arrange
    client = MagicMock(spec=httpx.Client)
    handler = _handler(client)
    handler.emit(_record("kept"))

    def _raise(pending: list) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(handler, "_push", _raise)

    # Act & Assert
    handler.flush()

    handler.close()
