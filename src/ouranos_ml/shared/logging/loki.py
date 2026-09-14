"""Batching handler that pushes structured log lines to a Grafana Loki gateway."""

import atexit
import json
import logging
import sys
import threading
from collections import deque

import httpx

_LOKI_PUSH_PATH = "/loki/api/v1/push"

_GRAFANA_LEVEL_NAMES: dict[int, str] = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warn",
    logging.ERROR: "error",
    logging.CRITICAL: "fatal",
}

_LEVEL_NAME_ALIASES: dict[str, str] = {"warning": "warn", "critical": "fatal"}


def to_grafana_level(levelno: int) -> str:
    """Map a stdlib log level number onto the Grafana level vocabulary."""
    for threshold in sorted(_GRAFANA_LEVEL_NAMES, reverse=True):
        if levelno >= threshold:
            return _GRAFANA_LEVEL_NAMES[threshold]
    return "debug"


def normalize_level_name(name: str) -> str:
    """Normalize a log level name onto the Grafana level vocabulary."""
    lowered = name.lower()
    return _LEVEL_NAME_ALIASES.get(lowered, lowered)


class LokiPushHandler(logging.Handler):
    """Logging handler that batches records and pushes them to a Loki gateway.

    Records are formatted on the emitting thread, buffered, and pushed from a
    background thread so that an unreachable Loki never blocks the application.
    Failed batches are dropped with a note on stderr rather than retried forever.
    """

    def __init__(
        self,
        base_url: str,
        tenant_id: str,
        app_name: str,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize the handler and start its background flush thread."""
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._tenant_id = tenant_id
        self._app_name = app_name
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=httpx.Timeout(5.0))
        self._buffer: deque[tuple[str, str, str]] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="loki-push", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer a formatted record; never raises."""
        try:
            line = self.format(record)
            timestamp_ns = str(int(record.created * 1e9))
            level = to_grafana_level(record.levelno)
            with self._lock:
                self._buffer.append((timestamp_ns, level, line))
                wake = len(self._buffer) >= self._batch_size
            if wake:
                self._wake.set()
        except Exception as exc:
            self._write_stderr(f"failed to buffer record: {exc}")

    def flush(self) -> None:
        """Push all buffered records to Loki, sorting values per stream."""
        with self._lock:
            pending = list(self._buffer)
            self._buffer.clear()
        if not pending:
            return
        try:
            self._push(pending)
        except Exception as exc:
            self._write_stderr(f"dropped {len(pending)} records: {exc}")

    def close(self) -> None:
        """Stop the background thread and push any remaining records; idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self._stop.set()
        self._wake.set()

        if self._thread.is_alive():
            self._thread.join(timeout=self._flush_interval + 5.0)

        self.flush()
        super().close()

        if self._owns_client:
            self._client.close()

    def _run(self) -> None:
        """Wake on the flush interval or a full batch and push the buffer."""
        while not self._stop.is_set():
            self._wake.wait(self._flush_interval)
            self._wake.clear()
            if self._stop.is_set():
                return
            self.flush()

    def _push(self, pending: list[tuple[str, str, str]]) -> None:
        values_by_level: dict[str, list[list[str]]] = {}
        for timestamp_ns, level, line in sorted(pending, key=lambda entry: entry[0]):
            timestamp_ns = self._ensure_strictly_increasing(timestamp_ns, values_by_level.get(level))
            values_by_level.setdefault(level, []).append([timestamp_ns, line])

        payload = json.dumps(
            {
                "streams": [
                    {"stream": {"app": self._app_name, "level": level}, "values": values}
                    for level, values in values_by_level.items()
                ]
            }
        )

        try:
            if self._post(payload).is_success:
                return
            error = "Loki returned an error response"
        except Exception as first_error:
            error = str(first_error)

        try:
            if self._post(payload).is_success:
                return
        except Exception:
            pass
        self._write_stderr(f"dropped {len(pending)} records: {error}")

    def _post(self, payload: str) -> httpx.Response:
        return self._client.post(
            f"{self._base_url}{_LOKI_PUSH_PATH}",
            content=payload,
            headers={"Content-Type": "application/json", "X-Scope-OrgID": self._tenant_id},
        )

    def _ensure_strictly_increasing(self, timestamp_ns: str, existing: list[list[str]] | None) -> str:
        if existing and int(timestamp_ns) <= int(existing[-1][0]):
            return str(int(existing[-1][0]) + 1)
        return timestamp_ns

    @staticmethod
    def _write_stderr(message: str) -> None:
        sys.stderr.write(f"[LokiPushHandler] {message}\n")
