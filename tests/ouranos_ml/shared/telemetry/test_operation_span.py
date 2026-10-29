import asyncio

import pytest
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from ouranos_ml.shared.telemetry import start_operation
from tests.ouranos_ml.shared.otel_mocks import patch_operation_telemetry


@pytest.fixture
def telemetry():
    """Record operation spans in memory and mock their duration histograms."""
    with patch_operation_telemetry() as recorded:
        yield recorded


def test_operation_span_when_ended_without_error_should_record_completed_span_and_duration(telemetry):
    # Arrange
    operation = start_operation("thing", attributes={"thing.input": "x"}, metric_attributes={"thing.kind": "a"})

    # Act
    operation.end()

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.name == "thing"
    assert dict(span.attributes) == {"thing.input": "x", "thing.kind": "a", "outcome": "completed"}
    assert span.status.status_code == StatusCode.UNSET
    telemetry.durations["thing"].record.assert_called_once()
    assert telemetry.durations["thing"].record.call_args.args[1] == {"thing.kind": "a", "outcome": "completed"}


def test_operation_span_when_ended_with_exception_should_record_failed_with_error_status(telemetry):
    # Arrange
    operation = start_operation("thing")

    # Act
    operation.end(ValueError("bad"))

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes["outcome"] == "failed"
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "ValueError: bad"
    assert [event.name for event in span.events] == ["exception"]
    assert telemetry.durations["thing"].record.call_args.args[1] == {"outcome": "failed"}


@pytest.mark.parametrize("error", [asyncio.CancelledError(), GeneratorExit()])
def test_operation_span_when_ended_with_base_exception_should_record_interrupted_without_error(telemetry, error):
    # Arrange
    operation = start_operation("thing")

    # Act
    operation.end(error)

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes["outcome"] == "interrupted"
    assert span.status.status_code == StatusCode.UNSET
    assert not span.events


def test_operation_span_when_record_metrics_disabled_should_end_span_without_duration(telemetry):
    # Arrange
    operation = start_operation("thing")

    # Act
    operation.end(ValueError("bad"), record_metrics=False)

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    telemetry.durations["thing"].record.assert_not_called()


def test_operation_span_when_ended_twice_should_record_once(telemetry):
    # Arrange
    operation = start_operation("thing")

    # Act
    operation.end()
    operation.end(ValueError("late"))

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes["outcome"] == "completed"
    telemetry.durations["thing"].record.assert_called_once()


def test_operation_span_when_metric_attributes_set_should_update_span_and_metrics(telemetry):
    # Arrange
    operation = start_operation("thing", metric_attributes={"thing.kind": "a"})

    # Act
    operation.set_metric_attributes({"thing.model": "m"})
    operation.set_attributes({"thing.count": 2})
    operation.add_event("halfway")
    operation.end()

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes["thing.model"] == "m"
    assert span.attributes["thing.count"] == 2
    assert [event.name for event in span.events] == ["halfway"]
    assert operation.metric_attributes == {"thing.kind": "a", "thing.model": "m"}
    assert telemetry.durations["thing"].record.call_args.args[1] == {
        "thing.kind": "a",
        "thing.model": "m",
        "outcome": "completed",
    }


def test_operation_span_when_activated_should_be_current_without_ending(telemetry):
    # Arrange
    operation = start_operation("thing")

    # Act
    with operation.activate():
        current = trace.get_current_span()

    # Assert
    assert current.get_span_context().span_id == operation._span.get_span_context().span_id
    assert trace.get_current_span() is not current
    assert not telemetry.spans.get_finished_spans()


def test_operation_span_when_used_as_context_manager_should_be_current_and_end_on_exit(telemetry):
    # Act
    with start_operation("thing") as operation:
        current = trace.get_current_span()

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert current.get_span_context().span_id == span.context.span_id
    assert span.attributes["outcome"] == "completed"
    assert operation.elapsed() >= 0


def test_operation_span_when_context_manager_raises_should_record_failed_and_reraise(telemetry):
    # Act & Assert
    with pytest.raises(RuntimeError), start_operation("thing"):
        raise RuntimeError("boom")

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes["outcome"] == "failed"
    assert [event.name for event in span.events] == ["exception"]
