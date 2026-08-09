import sys
from unittest.mock import MagicMock, patch

import pytest

from ouranos_ml.features.health.check.schemas import CheckStatus, HealthCheck, ServiceStatus
from ouranos_ml.features.health.check.service import CheckDef, check_gpu, check_llm_backend, handle


@pytest.mark.asyncio
async def test_check_llm_backend_when_url_empty_should_return_not_configured():
    # Arrange
    mock_settings = MagicMock()
    mock_settings.llm_openai_base_url = ""

    with patch("ouranos_ml.features.health.check.service.get_settings", return_value=mock_settings):
        # Act
        result = await check_llm_backend(timeout=5.0)

    # Assert
    assert result.status == CheckStatus.NOT_CONFIGURED
    assert "not configured" in result.description


@pytest.mark.asyncio
async def test_check_llm_backend_when_reachable_should_return_healthy():
    # Arrange
    mock_settings = MagicMock()
    mock_settings.llm_openai_base_url = "http://localhost:11434/v1"

    async def _model_generator():
        yield MagicMock()

    mock_client = MagicMock()
    mock_client.models.list.return_value = _model_generator()

    with (
        patch("ouranos_ml.features.health.check.service.get_settings", return_value=mock_settings),
        patch("ouranos_ml.features.health.check.service.get_openai_client", return_value=mock_client),
    ):
        # Act
        result = await check_llm_backend(timeout=5.0)

    # Assert
    assert result.status == CheckStatus.HEALTHY
    assert "reachable" in result.description


@pytest.mark.asyncio
async def test_check_llm_backend_when_timeout_should_return_unhealthy():
    # Arrange
    mock_settings = MagicMock()
    mock_settings.llm_openai_base_url = "http://localhost:11434/v1"
    mock_client = MagicMock()
    mock_client.models.list.return_value = MagicMock()

    with (
        patch("ouranos_ml.features.health.check.service.get_settings", return_value=mock_settings),
        patch("ouranos_ml.features.health.check.service.get_openai_client", return_value=mock_client),
        patch("ouranos_ml.features.health.check.service.asyncio.timeout", side_effect=TimeoutError),
    ):
        # Act
        result = await check_llm_backend(timeout=1.0)

    # Assert
    assert result.status == CheckStatus.UNHEALTHY
    assert "timed out" in result.description


@pytest.mark.asyncio
async def test_check_llm_backend_when_connection_fails_should_return_unhealthy():
    # Arrange
    mock_settings = MagicMock()
    mock_settings.llm_openai_base_url = "http://localhost:11434/v1"
    mock_client = MagicMock()
    mock_client.models.list.side_effect = ConnectionError("Connection refused")

    with (
        patch("ouranos_ml.features.health.check.service.get_settings", return_value=mock_settings),
        patch("ouranos_ml.features.health.check.service.get_openai_client", return_value=mock_client),
    ):
        # Act
        result = await check_llm_backend(timeout=5.0)

    # Assert
    assert result.status == CheckStatus.UNHEALTHY
    assert "unreachable" in result.description


def test_check_gpu_when_cuda_available_should_return_healthy():
    # Arrange
    torch = pytest.importorskip("torch", reason="torch not available in this environment", exc_type=ImportError)

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "device_count", return_value=2),
    ):
        # Act
        result = check_gpu()

    # Assert
    assert result.status == CheckStatus.HEALTHY
    assert "CUDA is available" in result.description
    assert result.data["deviceCount"] == 2


def test_check_gpu_when_cuda_unavailable_should_return_degraded():
    # Arrange
    torch = pytest.importorskip("torch", reason="torch not available in this environment", exc_type=ImportError)

    with patch.object(torch.cuda, "is_available", return_value=False):
        # Act
        result = check_gpu()

    # Assert
    assert result.status == CheckStatus.DEGRADED
    assert "CPU" in result.description


def test_check_gpu_when_torch_not_installed_should_return_not_configured(monkeypatch):
    # Arrange
    monkeypatch.setitem(sys.modules, "torch", None)

    # Act
    result = check_gpu()

    # Assert
    assert result.status == CheckStatus.NOT_CONFIGURED
    assert "PyTorch" in result.description


@pytest.mark.asyncio
async def test_handle_when_all_checks_healthy_should_return_healthy():
    # Arrange
    async def healthy_check(*args, **kwargs):
        return HealthCheck(status=CheckStatus.HEALTHY, description="ok")

    checks = [
        CheckDef(name="test_a", fn=healthy_check, needs_timeout=False),
        CheckDef(name="test_b", fn=healthy_check, needs_timeout=False),
    ]

    with patch("ouranos_ml.features.health.check.service._CHECKS", checks):
        # Act
        result = await handle()

    # Assert
    assert result.status == ServiceStatus.HEALTHY


@pytest.mark.asyncio
async def test_handle_when_one_check_unhealthy_should_return_unhealthy():
    # Arrange
    async def healthy_check(*args, **kwargs):
        return HealthCheck(status=CheckStatus.HEALTHY, description="ok")

    async def unhealthy_check(*args, **kwargs):
        return HealthCheck(status=CheckStatus.UNHEALTHY, description="bad")

    checks = [
        CheckDef(name="test_a", fn=healthy_check, needs_timeout=False),
        CheckDef(name="test_b", fn=unhealthy_check, needs_timeout=False),
    ]

    with patch("ouranos_ml.features.health.check.service._CHECKS", checks):
        # Act
        result = await handle()

    # Assert
    assert result.status == ServiceStatus.UNHEALTHY
    assert result.checks["test_b"].status == CheckStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_handle_when_one_check_degraded_none_unhealthy_should_return_degraded():
    # Arrange
    async def healthy_check(*args, **kwargs):
        return HealthCheck(status=CheckStatus.HEALTHY, description="ok")

    async def degraded_check(*args, **kwargs):
        return HealthCheck(status=CheckStatus.DEGRADED, description="degraded")

    checks = [
        CheckDef(name="test_a", fn=healthy_check, needs_timeout=False),
        CheckDef(name="test_b", fn=degraded_check, needs_timeout=False),
    ]

    with patch("ouranos_ml.features.health.check.service._CHECKS", checks):
        # Act
        result = await handle()

    # Assert
    assert result.status == ServiceStatus.DEGRADED


@pytest.mark.asyncio
async def test_handle_when_all_checks_not_configured_should_return_not_configured():
    # Arrange
    async def not_configured_check(*args, **kwargs):
        return HealthCheck(status=CheckStatus.NOT_CONFIGURED, description="not configured")

    checks = [
        CheckDef(name="test_a", fn=not_configured_check, needs_timeout=False),
        CheckDef(name="test_b", fn=not_configured_check, needs_timeout=False),
    ]

    with patch("ouranos_ml.features.health.check.service._CHECKS", checks):
        # Act
        result = await handle()

    # Assert
    assert result.status == ServiceStatus.NOT_CONFIGURED


@pytest.mark.asyncio
async def test_handle_when_check_raises_exception_should_return_unhealthy():
    # Arrange
    async def healthy_check(*args, **kwargs):
        return HealthCheck(status=CheckStatus.HEALTHY, description="ok")

    async def raising_check(*args, **kwargs):
        raise RuntimeError("boom")

    checks = [
        CheckDef(name="test_a", fn=healthy_check, needs_timeout=False),
        CheckDef(name="test_b", fn=raising_check, needs_timeout=False),
    ]

    with patch("ouranos_ml.features.health.check.service._CHECKS", checks):
        # Act
        result = await handle()

    # Assert
    assert result.status == ServiceStatus.UNHEALTHY
    assert result.checks["test_b"].status == CheckStatus.UNHEALTHY
    assert result.checks["test_b"].description == "boom"
