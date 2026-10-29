"""Shared helpers for asserting on OpenTelemetry spans and metrics without a global provider."""

from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from ouranos_ml.shared.telemetry import operation_span


def make_recording_tracer() -> tuple[Tracer, InMemorySpanExporter]:
    """Create a tracer whose finished spans are collected by the returned exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("tests"), exporter


@dataclass
class OperationTelemetry:
    """What ``OperationSpan``s recorded: finished spans, and a mock ``<name>.duration`` histogram per name."""

    spans: InMemorySpanExporter
    durations: defaultdict[str, MagicMock]


@contextmanager
def patch_operation_telemetry() -> Generator[OperationTelemetry]:
    """Record ``OperationSpan`` spans in memory and replace their duration histograms with mocks."""
    tracer, spans = make_recording_tracer()
    durations: defaultdict[str, MagicMock] = defaultdict(MagicMock)
    with (
        patch.object(operation_span, "_tracer", tracer),
        patch.object(operation_span, "_duration_histogram", side_effect=lambda name: durations[name]),
    ):
        yield OperationTelemetry(spans=spans, durations=durations)
