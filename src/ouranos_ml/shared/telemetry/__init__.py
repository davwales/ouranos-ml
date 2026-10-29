"""Manual OpenTelemetry instrumentation for operations that no installed instrumentation covers."""

from ouranos_ml.shared.telemetry.operation_span import OperationSpan, start_operation

__all__ = ["OperationSpan", "start_operation"]
