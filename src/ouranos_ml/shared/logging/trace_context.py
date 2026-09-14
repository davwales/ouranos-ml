"""Structlog processor that injects OpenTelemetry trace context into log bodies."""

from opentelemetry import trace
from structlog.typing import EventDict, FilteringBoundLogger


def add_trace_context(_logger: FilteringBoundLogger, _method_name: str, event_dict: EventDict) -> EventDict:
    """Add TraceId and SpanId to a log event when a recording span is active.

    Keys use the PascalCase shape expected by the Grafana Loki derived field
    that links log lines to Tempo traces ("TraceId":"<32 hex>"), matching the
    body shape produced by the pantheon gateway's Serilog pipeline.
    """
    span = trace.get_current_span()
    context = span.get_span_context()
    if span.is_recording() and context.is_valid:
        event_dict["TraceId"] = format(context.trace_id, "032x")
        event_dict["SpanId"] = format(context.span_id, "016x")
    return event_dict
