"""Shared helpers for asserting on OpenTelemetry spans without a global provider."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer


def make_recording_tracer() -> tuple[Tracer, InMemorySpanExporter]:
    """Create a tracer whose finished spans are collected by the returned exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("tests"), exporter
