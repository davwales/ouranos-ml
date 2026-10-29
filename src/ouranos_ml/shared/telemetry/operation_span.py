"""A span with a built-in duration metric and outcome, for operations no instrumentation covers."""

import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import Token
from functools import lru_cache
from types import TracebackType
from typing import Self

from opentelemetry import context, metrics, trace
from opentelemetry.context import Context
from opentelemetry.metrics import Histogram
from opentelemetry.util.types import AttributeValue

_tracer = trace.get_tracer(__name__)
_meter = metrics.get_meter(__name__)


@lru_cache
def _duration_histogram(name: str) -> Histogram:
    """The ``<name>.duration`` histogram, created once per operation name."""
    return _meter.create_histogram(f"{name}.duration", unit="s", description=f"Duration of {name} operations")


class OperationSpan:
    """A span that records ``<name>.duration`` and an ``outcome`` when it ends.

    Span attributes may be high cardinality; metric attributes are set on the span too and become the metric's
    dimensions, so keep them low cardinality. Use it as a context manager for a scoped operation, or call
    ``activate()`` and ``end()`` separately when the operation starts in one call and finishes in another.

    The outcome is ``completed`` without an error, ``failed`` (with the exception recorded and ERROR status) for an
    ``Exception``, and ``interrupted`` for any other ``BaseException``, such as a client disconnect.
    """

    def __init__(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
        metric_attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        self._name = name
        self._metric_attributes = dict(metric_attributes or {})
        self._span = _tracer.start_span(name, attributes={**(attributes or {}), **self._metric_attributes})
        self._started = time.monotonic()
        self._ended = False
        self._token: Token[Context] | None = None

    @property
    def metric_attributes(self) -> dict[str, AttributeValue]:
        """The metric dimensions, for feature-specific instruments that should match the duration metric."""
        return dict(self._metric_attributes)

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        """Set span-only attributes."""
        self._span.set_attributes(attributes)

    def set_metric_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        """Set attributes on the span that are also metric dimensions."""
        self._metric_attributes.update(attributes)
        self._span.set_attributes(attributes)

    def add_event(self, name: str, attributes: Mapping[str, AttributeValue] | None = None) -> None:
        """Add a timestamped event to the span."""
        self._span.add_event(name, attributes)

    def elapsed(self) -> float:
        """Seconds since the operation started."""
        return time.monotonic() - self._started

    @contextmanager
    def activate(self) -> Generator[Self]:
        """Make the span current, so spans started inside nest under it, without ending it."""
        with trace.use_span(self._span, end_on_exit=False, record_exception=False, set_status_on_exception=False):
            yield self

    def end(self, error: BaseException | None = None, *, record_metrics: bool = True) -> None:
        """Record the outcome on the span and, unless disabled, the duration metric, then end the span (once)."""
        if self._ended:
            return
        self._ended = True

        if error is None:
            outcome = "completed"
        elif isinstance(error, Exception):
            outcome = "failed"
            self._span.record_exception(error)
            self._span.set_status(trace.StatusCode.ERROR, f"{type(error).__name__}: {error}")
        else:
            outcome = "interrupted"

        self._span.set_attribute("outcome", outcome)
        if record_metrics:
            _duration_histogram(self._name).record(self.elapsed(), {**self._metric_attributes, "outcome": outcome})
        self._span.end()

    def __enter__(self) -> Self:
        self._token = context.attach(trace.set_span_in_context(self._span))
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if self._token is not None:
            context.detach(self._token)
        self.end(exc)


def start_operation(
    name: str,
    *,
    attributes: Mapping[str, AttributeValue] | None = None,
    metric_attributes: Mapping[str, AttributeValue] | None = None,
) -> OperationSpan:
    """Start an operation span; see ``OperationSpan``."""
    return OperationSpan(name, attributes, metric_attributes)
