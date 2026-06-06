import asyncio
import logging
from collections.abc import Callable
from typing import Any, NamedTuple

from ouranos_ml.features.health.check.schemas import (
    CheckStatus,
    GpuCheckData,
    HealthCheck,
    HealthResponse,
    ServiceStatus,
)
from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.infra.clients.llm_client import get_openai_client

logger = logging.getLogger(__name__)


class CheckDef(NamedTuple):
    """Definition of a single health check in the registry."""

    name: str
    fn: Callable[..., Any]
    needs_timeout: bool


async def check_llm_backend(timeout: float) -> HealthCheck:
    """Check if the LLM backend is reachable."""
    settings = get_settings()
    if not settings.llm_openai_base_url:
        return HealthCheck(status=CheckStatus.NOT_CONFIGURED, description="LLM backend is not configured")

    try:
        client = get_openai_client()
        async with asyncio.timeout(timeout):
            async for _ in client.models.list():
                break
        return HealthCheck(status=CheckStatus.HEALTHY, description="LLM backend is reachable")
    except TimeoutError:
        return HealthCheck(status=CheckStatus.UNHEALTHY, description=f"LLM backend timed out after {timeout}s")
    except Exception:
        logger.exception("LLM backend health check failed")
        return HealthCheck(status=CheckStatus.UNHEALTHY, description="LLM backend is unreachable")


def check_gpu() -> HealthCheck:
    """Check if CUDA GPU is available."""
    try:
        import torch
    except ImportError:
        return HealthCheck(status=CheckStatus.NOT_CONFIGURED, description="PyTorch is not installed")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        gpu_data = GpuCheckData(device_count=device_count)
        return HealthCheck(
            status=CheckStatus.HEALTHY,
            description=f"CUDA is available ({device_count} device(s))",
            data=gpu_data.model_dump(by_alias=True),
        )

    return HealthCheck(status=CheckStatus.DEGRADED, description="CUDA not available — running on CPU")


_CHECKS: list[CheckDef] = [
    CheckDef(name="llm", fn=check_llm_backend, needs_timeout=True),
    CheckDef(name="gpu", fn=check_gpu, needs_timeout=False),
]


async def handle() -> HealthResponse:
    """Run all health checks concurrently and compute aggregate status."""
    settings = get_settings()
    timeout = settings.health_check_timeout_seconds

    async def _run_check(entry: CheckDef) -> tuple[str, HealthCheck]:
        """Run a single check, catching any unhandled exceptions."""
        try:
            if entry.needs_timeout:
                result = await entry.fn(timeout)
            elif asyncio.iscoroutinefunction(entry.fn):
                result = await entry.fn()
            else:
                result = await asyncio.to_thread(entry.fn)
        except Exception as exc:
            result = HealthCheck(status=CheckStatus.UNHEALTHY, description=str(exc))
        return entry.name, result

    results = await asyncio.gather(*(_run_check(entry) for entry in _CHECKS))
    checks = dict(results)

    if not checks:
        status = ServiceStatus.HEALTHY
    elif any(c.status == CheckStatus.UNHEALTHY for c in checks.values()):
        status = ServiceStatus.UNHEALTHY
    elif any(c.status == CheckStatus.DEGRADED for c in checks.values()):
        status = ServiceStatus.DEGRADED
    elif all(c.status == CheckStatus.NOT_CONFIGURED for c in checks.values()):
        status = ServiceStatus.NOT_CONFIGURED
    else:
        status = ServiceStatus.HEALTHY

    return HealthResponse(status=status, checks=checks)