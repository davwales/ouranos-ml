"""Process-wide structured logging configuration (console + optional Loki sink)."""

import logging
import sys

import structlog
from structlog.typing import EventDict, FilteringBoundLogger, Processor

from ouranos_ml.shared.domain.core.settings import Settings
from ouranos_ml.shared.logging.loki import LokiPushHandler, normalize_level_name
from ouranos_ml.shared.logging.trace_context import add_trace_context

_NOISY_LOGGERS: tuple[str, ...] = ("httpx", "openai", "matplotlib")


class ConsoleHandler(logging.StreamHandler):
    """Stream handler installed by configure_logging on stdout."""


def configure_logging(settings: Settings) -> None:
    """Configure the process-wide structured logging pipeline.

    Wires structlog and stdlib logging through a shared processor chain, renders
    either a pretty console (local dev) or JSON (production), and attaches the
    Loki push handler when a Loki base URL is configured.
    """
    level = logging.getLevelName(settings.log_level.upper())
    if not isinstance(level, int):
        raise ValueError(f"Invalid LOG_LEVEL '{settings.log_level}'")
    use_json = settings.log_json if settings.log_json is not None else bool(settings.loki_base_url)

    pre_chain = _build_pre_chain()
    structlog.configure(
        processors=[*pre_chain, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    root.setLevel(level)
    _remove_installed_handlers(root)

    console = ConsoleHandler(sys.stdout)
    console.setFormatter(
        _build_formatter(
            _json_renderer() if use_json else structlog.dev.ConsoleRenderer(),
            pre_chain,
        )
    )
    console.setLevel(logging.WARNING if settings.loki_base_url else level)
    root.addHandler(console)

    if settings.loki_base_url:
        loki = LokiPushHandler(
            base_url=settings.loki_base_url,
            tenant_id=settings.loki_tenant_id,
            app_name=settings.log_app_name,
        )
        loki.setFormatter(_build_formatter(_json_renderer(), pre_chain))
        root.addHandler(loki)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _json_renderer() -> structlog.processors.JSONRenderer:
    """Build the compact JSON renderer shared by console and Loki output.

    Compact separators keep rendered lines shape-compatible with the Grafana
    Loki derived field that links "TraceId" body keys to Tempo traces.
    """
    return structlog.processors.JSONRenderer(separators=(",", ":"))


def _build_pre_chain() -> list[Processor]:
    """Build the processor chain shared by structlog and stdlib log records."""
    return [
        structlog.contextvars.merge_contextvars,
        add_trace_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _normalize_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.EventRenamer(to="message"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


def _normalize_level(_logger: FilteringBoundLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Rewrite the level field onto the Grafana level vocabulary."""
    if "level" in event_dict:
        event_dict["level"] = normalize_level_name(event_dict["level"])
    return event_dict


def _build_formatter(renderer: Processor, pre_chain: list[Processor]) -> structlog.stdlib.ProcessorFormatter:
    """Build a ProcessorFormatter that renders both native and foreign records."""
    return structlog.stdlib.ProcessorFormatter(
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        foreign_pre_chain=pre_chain,
    )


def _remove_installed_handlers(root: logging.Logger) -> None:
    """Remove only handlers previously installed by this module, keeping foreign ones."""
    installed = (handler for handler in list(root.handlers) if isinstance(handler, (ConsoleHandler, LokiPushHandler)))
    for handler in installed:
        root.removeHandler(handler)
        handler.close()
