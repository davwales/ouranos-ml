import asyncio
from collections.abc import Awaitable, Callable

import structlog

from ouranos_ml.features.health.check.schemas import (
    GpuCheckData,
    HealthCheck,
    HealthResponse,
    HealthStatus,
    PlutusCheckData,
    SwarmUICheckData,
)
from ouranos_ml.features.plutus.forecast.service import get_forecast_generator
from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.infra.clients.llm_client import get_openai_client
from ouranos_ml.shared.infra.clients.swarmui_client import get_swarmui_client

logger = structlog.get_logger(__name__)

# Checks whose last run failed. pantheon polls /health continuously, so failures are logged on the transition
# into the failed state (and recoveries on the way out) rather than on every poll during an outage.
_failing_checks: set[str] = set()


def _log_failure(check: str, exc: Exception) -> None:
    """Warn, without a traceback, the first time a check fails after being healthy."""
    if check in _failing_checks:
        return
    _failing_checks.add(check)
    logger.warning("health check failed", check=check, error_type=type(exc).__name__, error=str(exc))


def _log_recovery(check: str) -> None:
    """Log when a previously failing check succeeds again."""
    if check not in _failing_checks:
        return
    _failing_checks.discard(check)
    logger.info("health check recovered", check=check)


async def check_llm_backend(timeout: float) -> HealthCheck:
    """Check if the LLM backend is reachable."""
    settings = get_settings()
    if not settings.llm_openai_base_url:
        return HealthCheck(status=HealthStatus.NOT_CONFIGURED, description="LLM backend is not configured")

    try:
        client = get_openai_client()
        async with asyncio.timeout(timeout):
            async for _ in client.models.list():
                break
        _log_recovery("llm")
        return HealthCheck(status=HealthStatus.HEALTHY, description="LLM backend is reachable")
    except TimeoutError:
        return HealthCheck(status=HealthStatus.UNHEALTHY, description=f"LLM backend timed out after {timeout}s")
    except Exception as exc:
        _log_failure("llm", exc)
        return HealthCheck(status=HealthStatus.UNHEALTHY, description="LLM backend is unreachable")


async def check_swarmui(timeout: float) -> HealthCheck:
    """Check if SwarmUI (image generation) is reachable and its backends are usable.

    Failures are DEGRADED rather than UNHEALTHY: image generation being down should not mark the whole service
    unhealthy for callers that only use the LLM features.
    """
    settings = get_settings()
    if not settings.swarmui_base_url:
        return HealthCheck(status=HealthStatus.NOT_CONFIGURED, description="SwarmUI is not configured")

    try:
        client = get_swarmui_client()
        async with asyncio.timeout(timeout):
            session_id = await client.new_session()
            response = await client.call("GetCurrentStatus", session_id)
    except TimeoutError:
        return HealthCheck(status=HealthStatus.DEGRADED, description=f"SwarmUI timed out after {timeout}s")
    except Exception as exc:
        _log_failure("swarmui", exc)
        return HealthCheck(status=HealthStatus.DEGRADED, description="SwarmUI is unreachable")

    _log_recovery("swarmui")
    backend_status = response.get("backend_status", {}).get("status", "unknown")
    data = SwarmUICheckData(backend_status=backend_status).model_dump(by_alias=True)
    if backend_status in ("errored", "all_disabled", "empty"):
        return HealthCheck(
            status=HealthStatus.DEGRADED, description=f"SwarmUI backends are {backend_status}", data=data
        )
    return HealthCheck(status=HealthStatus.HEALTHY, description="SwarmUI is reachable", data=data)


def check_gpu() -> HealthCheck:
    """Check if CUDA GPU is available."""
    try:
        import torch
    except ImportError:
        return HealthCheck(status=HealthStatus.NOT_CONFIGURED, description="PyTorch is not installed")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        gpu_data = GpuCheckData(device_count=device_count)
        return HealthCheck(
            status=HealthStatus.HEALTHY,
            description=f"CUDA is available ({device_count} device(s))",
            data=gpu_data.model_dump(by_alias=True),
        )

    return HealthCheck(status=HealthStatus.DEGRADED, description="CUDA not available — running on CPU")


def check_plutus() -> HealthCheck:
    """Check if the Plutus forecast model artifact loads.

    Loading goes through the cached forecast generator, so the first probe pays the model load (and warms it for
    forecast requests). Missing artifacts are NOT_CONFIGURED; other load failures are DEGRADED rather than UNHEALTHY,
    for the same reason as SwarmUI.
    """
    settings = get_settings()
    try:
        generator = get_forecast_generator()
    except FileNotFoundError:
        return HealthCheck(status=HealthStatus.NOT_CONFIGURED, description="Plutus model artifacts not found")
    except Exception as exc:
        _log_failure("plutus", exc)
        return HealthCheck(status=HealthStatus.DEGRADED, description="Plutus model failed to load")

    _log_recovery("plutus")

    data = PlutusCheckData(model_name=settings.plutus_forecast_model_name, sequence_length=generator.sequence_length)
    return HealthCheck(
        status=HealthStatus.HEALTHY, description="Plutus model is loaded", data=data.model_dump(by_alias=True)
    )


type _Check = Callable[[float], Awaitable[HealthCheck]]


def _in_thread(check: Callable[[], HealthCheck]) -> _Check:
    """Adapt a blocking check to the registry signature; it runs in a worker thread and ignores the timeout."""

    async def run(_timeout: float) -> HealthCheck:
        return await asyncio.to_thread(check)

    return run


_CHECKS: dict[str, _Check] = {
    "llm": check_llm_backend,
    "gpu": _in_thread(check_gpu),
    "swarmui": check_swarmui,
    "plutus": _in_thread(check_plutus),
}


async def handle() -> HealthResponse:
    """Run all health checks concurrently and compute aggregate status."""
    settings = get_settings()
    timeout = settings.health_check_timeout_seconds

    async def _run_check(name: str, check: _Check) -> tuple[str, HealthCheck]:
        """Run a single check, catching any unhandled exceptions."""
        try:
            result = await check(timeout)
        except Exception as exc:
            result = HealthCheck(status=HealthStatus.UNHEALTHY, description=str(exc))
        return name, result

    results = await asyncio.gather(*(_run_check(name, check) for name, check in _CHECKS.items()))
    checks = dict(results)

    for name, check in checks.items():
        logger.debug("health check complete", check=name, status=check.status.value)

    if not checks:
        status = HealthStatus.HEALTHY
    elif any(c.status == HealthStatus.UNHEALTHY for c in checks.values()):
        status = HealthStatus.UNHEALTHY
    elif any(c.status == HealthStatus.DEGRADED for c in checks.values()):
        status = HealthStatus.DEGRADED
    elif all(c.status == HealthStatus.NOT_CONFIGURED for c in checks.values()):
        status = HealthStatus.NOT_CONFIGURED
    else:
        status = HealthStatus.HEALTHY

    return HealthResponse(status=status, checks=checks)
