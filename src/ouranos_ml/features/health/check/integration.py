import asyncio
import logging

from ouranos_ml.features.health.check.schemas import CheckStatus, GpuCheckData, HealthCheck
from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.infra.clients.llm_client import get_openai_client

logger = logging.getLogger(__name__)


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
