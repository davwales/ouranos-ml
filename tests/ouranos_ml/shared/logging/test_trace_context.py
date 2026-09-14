"""Tests for ouranos_ml.shared.logging.trace_context."""

import json
import re

import pytest
import structlog
from opentelemetry.context import attach, detach
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import NonRecordingSpan, SpanContext, set_span_in_context

from ouranos_ml.shared.domain.core.settings import Settings
from ouranos_ml.shared.logging.config import configure_logging
from ouranos_ml.shared.logging.trace_context import add_trace_context

_GRAFANA_TRACE_ID_REGEX = r'"TraceId":"([0-9a-f]{32})"'


def _settings() -> Settings:
    """Build settings for trace context processor tests."""
    return Settings(log_level="INFO", log_json=True)


def _recording_tracer() -> tuple[TracerProvider, object]:
    """Build a local tracer provider so the process-wide provider is untouched."""
    provider = TracerProvider()
    return provider, provider.get_tracer("trace-context-tests")


def test_add_trace_context_when_span_recording_should_add_trace_and_span_ids() -> None:
    """Test that a recording span injects hex TraceId and SpanId keys."""
    # Arrange
    _, tracer = _recording_tracer()
    event_dict: dict = {"message": "inside request"}

    # Act
    with tracer.start_as_current_span("test-span") as span:
        result = add_trace_context(None, "info", event_dict)
        span_context = span.get_span_context()

    # Assert
    assert result["TraceId"] == format(span_context.trace_id, "032x")
    assert result["SpanId"] == format(span_context.span_id, "016x")
    assert re.fullmatch(r"[0-9a-f]{32}", result["TraceId"])
    assert re.fullmatch(r"[0-9a-f]{16}", result["SpanId"])


def test_add_trace_context_when_no_span_should_not_add_keys() -> None:
    """Test that an event outside any span stays untouched."""
    # Arrange
    event_dict: dict = {"message": "outside request"}

    # Act
    result = add_trace_context(None, "info", event_dict)

    # Assert
    assert "TraceId" not in result
    assert "SpanId" not in result


def test_add_trace_context_when_span_not_recording_should_not_add_keys() -> None:
    """Test that a non-recording span in context adds no keys."""
    # Arrange
    invalid_context = SpanContext(trace_id=1, span_id=1, is_remote=False)
    token = attach(set_span_in_context(NonRecordingSpan(invalid_context)))
    event_dict: dict = {"message": "not recorded"}

    try:
        # Act
        result = add_trace_context(None, "info", event_dict)
    finally:
        detach(token)

    # Assert
    assert "TraceId" not in result
    assert "SpanId" not in result


def test_add_trace_context_when_json_rendered_should_match_grafana_derived_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that rendered JSON lines carry the TraceId shape Loki links to Tempo."""
    # Arrange
    configure_logging(_settings())
    _, tracer = _recording_tracer()

    # Act
    with tracer.start_as_current_span("test-span"):
        structlog.get_logger("test.trace").info("inside span")

    # Assert
    line = capsys.readouterr().out.strip().splitlines()[-1]
    match = re.search(_GRAFANA_TRACE_ID_REGEX, line)
    assert match is not None
    parsed = json.loads(line)
    assert parsed["message"] == "inside span"
    assert parsed["TraceId"] == match.group(1)


def test_add_trace_context_when_no_span_should_render_without_trace_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that rendered JSON lines outside spans omit the trace keys."""
    # Arrange
    configure_logging(_settings())

    # Act
    structlog.get_logger("test.trace").info("outside span")

    # Assert
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert '"TraceId"' not in line
    assert '"SpanId"' not in line
