import asyncio

from ouranos_ml.features.health.check.integration import check_gpu, check_llm_backend
from ouranos_ml.features.health.check.schemas import CheckDef, CheckStatus, HealthCheck, HealthResponse, ServiceStatus
from ouranos_ml.shared.domain.core.settings import get_settings

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
