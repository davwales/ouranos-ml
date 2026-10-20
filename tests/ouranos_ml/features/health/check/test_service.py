import sys
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

from ouranos_ml.features.health.check import service
from ouranos_ml.features.health.check.schemas import HealthCheck, HealthStatus
from ouranos_ml.features.health.check.service import (
    check_gpu,
    check_llm_backend,
    check_plutus,
    check_swarmui,
    handle,
)
from ouranos_ml.shared.infra.swarmui.errors import SwarmUIUnavailableError

SWARMUI_SETTINGS = "ouranos_ml.features.health.check.service.get_settings"
SWARMUI_CLIENT = "ouranos_ml.features.health.check.service.get_swarmui_client"
LLM_SETTINGS = "ouranos_ml.features.health.check.service.get_settings"
LLM_CLIENT = "ouranos_ml.features.health.check.service.get_openai_client"


@pytest.fixture(autouse=True)
def reset_failing_checks():
    service._failing_checks.clear()
    yield
    service._failing_checks.clear()


def _llm_client(*, fail: bool) -> MagicMock:
    async def _model_generator():
        yield MagicMock()

    client = MagicMock()
    if fail:
        client.models.list.side_effect = ConnectionError("Connection refused")
    else:
        client.models.list.return_value = _model_generator()
    return client


FORECAST_GENERATOR = "ouranos_ml.features.health.check.service.get_forecast_generator"


@pytest.mark.asyncio
async def test_check_llm_backend_when_url_empty_should_return_not_configured():
    # Arrange
    mock_settings = MagicMock()
    mock_settings.llm_openai_base_url = ""

    with patch("ouranos_ml.features.health.check.service.get_settings", return_value=mock_settings):
        # Act
        result = await check_llm_backend(timeout=5.0)

    # Assert
    assert result.status == HealthStatus.NOT_CONFIGURED
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
    assert result.status == HealthStatus.HEALTHY
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
    assert result.status == HealthStatus.UNHEALTHY
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
    assert result.status == HealthStatus.UNHEALTHY
    assert "unreachable" in result.description


@pytest.mark.asyncio
async def test_check_llm_backend_when_connection_fails_should_warn_without_traceback():
    # Arrange
    mock_settings = MagicMock(llm_openai_base_url="http://localhost:11434/v1")

    with (
        patch(LLM_SETTINGS, return_value=mock_settings),
        patch(LLM_CLIENT, return_value=_llm_client(fail=True)),
        capture_logs() as logs,
    ):
        # Act
        await check_llm_backend(timeout=5.0)

    # Assert
    assert logs == [
        {
            "event": "health check failed",
            "log_level": "warning",
            "check": "llm",
            "error_type": "ConnectionError",
            "error": "Connection refused",
        }
    ]


@pytest.mark.asyncio
async def test_check_llm_backend_when_failing_repeatedly_should_warn_once():
    # Arrange
    mock_settings = MagicMock(llm_openai_base_url="http://localhost:11434/v1")

    with (
        patch(LLM_SETTINGS, return_value=mock_settings),
        patch(LLM_CLIENT, return_value=_llm_client(fail=True)),
        capture_logs() as logs,
    ):
        # Act
        await check_llm_backend(timeout=5.0)
        await check_llm_backend(timeout=5.0)

    # Assert
    assert [log["log_level"] for log in logs] == ["warning"]


@pytest.mark.asyncio
async def test_check_llm_backend_when_recovering_after_failure_should_log_info():
    # Arrange
    mock_settings = MagicMock(llm_openai_base_url="http://localhost:11434/v1")

    with patch(LLM_SETTINGS, return_value=mock_settings), capture_logs() as logs:
        with patch(LLM_CLIENT, return_value=_llm_client(fail=True)):
            await check_llm_backend(timeout=5.0)

        with patch(LLM_CLIENT, return_value=_llm_client(fail=False)):
            # Act
            await check_llm_backend(timeout=5.0)
            await check_llm_backend(timeout=5.0)

    # Assert
    assert logs[1:] == [{"event": "health check recovered", "log_level": "info", "check": "llm"}]


@pytest.mark.asyncio
async def test_check_llm_backend_when_healthy_without_prior_failure_should_not_log():
    # Arrange
    mock_settings = MagicMock(llm_openai_base_url="http://localhost:11434/v1")

    with (
        patch(LLM_SETTINGS, return_value=mock_settings),
        patch(LLM_CLIENT, return_value=_llm_client(fail=False)),
        capture_logs() as logs,
    ):
        # Act
        await check_llm_backend(timeout=5.0)

    # Assert
    assert logs == []


def _swarmui_client(backend_status: str = "running") -> MagicMock:
    client = MagicMock()
    client.new_session = AsyncMock(return_value="session-1")
    client.call = AsyncMock(return_value={"status": {}, "backend_status": {"status": backend_status}})
    return client


@pytest.mark.asyncio
async def test_check_swarmui_when_url_empty_should_return_not_configured():
    # Arrange
    mock_settings = MagicMock(swarmui_base_url="")

    with patch(SWARMUI_SETTINGS, return_value=mock_settings):
        # Act
        result = await check_swarmui(timeout=5.0)

    # Assert
    assert result.status == HealthStatus.NOT_CONFIGURED


@pytest.mark.asyncio
async def test_check_swarmui_when_reachable_and_running_should_return_healthy_with_backend_status():
    # Arrange
    mock_settings = MagicMock(swarmui_base_url="http://swarm:7801")
    client = _swarmui_client()

    with patch(SWARMUI_SETTINGS, return_value=mock_settings), patch(SWARMUI_CLIENT, return_value=client):
        # Act
        result = await check_swarmui(timeout=5.0)

    # Assert
    assert result.status == HealthStatus.HEALTHY
    assert result.data == {"backend_status": "running"}
    client.call.assert_awaited_once_with("GetCurrentStatus", "session-1")


@pytest.mark.asyncio
async def test_check_swarmui_when_backends_errored_should_return_degraded():
    # Arrange
    mock_settings = MagicMock(swarmui_base_url="http://swarm:7801")

    with (
        patch(SWARMUI_SETTINGS, return_value=mock_settings),
        patch(SWARMUI_CLIENT, return_value=_swarmui_client("errored")),
    ):
        # Act
        result = await check_swarmui(timeout=5.0)

    # Assert
    assert result.status == HealthStatus.DEGRADED
    assert "errored" in result.description


@pytest.mark.asyncio
async def test_check_swarmui_when_unreachable_should_return_degraded_not_unhealthy():
    # Arrange
    mock_settings = MagicMock(swarmui_base_url="http://swarm:7801")
    client = _swarmui_client()
    client.new_session.side_effect = SwarmUIUnavailableError("down")

    with patch(SWARMUI_SETTINGS, return_value=mock_settings), patch(SWARMUI_CLIENT, return_value=client):
        # Act
        result = await check_swarmui(timeout=5.0)

    # Assert
    assert result.status == HealthStatus.DEGRADED
    assert "unreachable" in result.description


@pytest.mark.asyncio
async def test_check_swarmui_when_unreachable_should_warn_without_traceback():
    # Arrange
    mock_settings = MagicMock(swarmui_base_url="http://swarm:7801")
    client = _swarmui_client()
    client.new_session.side_effect = SwarmUIUnavailableError("down")

    with (
        patch(SWARMUI_SETTINGS, return_value=mock_settings),
        patch(SWARMUI_CLIENT, return_value=client),
        capture_logs() as logs,
    ):
        # Act
        await check_swarmui(timeout=5.0)
        await check_swarmui(timeout=5.0)

    # Assert
    assert logs == [
        {
            "event": "health check failed",
            "log_level": "warning",
            "check": "swarmui",
            "error_type": "SwarmUIUnavailableError",
            "error": "down",
        }
    ]


@pytest.mark.asyncio
async def test_check_swarmui_when_recovering_after_failure_should_log_info():
    # Arrange
    mock_settings = MagicMock(swarmui_base_url="http://swarm:7801")
    client = _swarmui_client()
    client.new_session.side_effect = SwarmUIUnavailableError("down")

    with patch(SWARMUI_SETTINGS, return_value=mock_settings), patch(SWARMUI_CLIENT, return_value=client):
        with capture_logs() as logs:
            await check_swarmui(timeout=5.0)
            client.new_session.side_effect = None

            # Act
            await check_swarmui(timeout=5.0)

    # Assert
    assert logs[1:] == [{"event": "health check recovered", "log_level": "info", "check": "swarmui"}]


@pytest.mark.asyncio
async def test_check_swarmui_when_timeout_should_return_degraded():
    # Arrange
    mock_settings = MagicMock(swarmui_base_url="http://swarm:7801")

    with (
        patch(SWARMUI_SETTINGS, return_value=mock_settings),
        patch(SWARMUI_CLIENT, return_value=_swarmui_client()),
        patch("ouranos_ml.features.health.check.service.asyncio.timeout", side_effect=TimeoutError),
    ):
        # Act
        result = await check_swarmui(timeout=5.0)

    # Assert
    assert result.status == HealthStatus.DEGRADED
    assert "timed out" in result.description


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
    assert result.status == HealthStatus.HEALTHY
    assert "CUDA is available" in result.description
    assert result.data["device_count"] == 2


def test_check_gpu_when_cuda_unavailable_should_return_degraded():
    # Arrange
    torch = pytest.importorskip("torch", reason="torch not available in this environment", exc_type=ImportError)

    with patch.object(torch.cuda, "is_available", return_value=False):
        # Act
        result = check_gpu()

    # Assert
    assert result.status == HealthStatus.DEGRADED
    assert "CPU" in result.description


def test_check_gpu_when_torch_not_installed_should_return_not_configured(monkeypatch):
    # Arrange
    monkeypatch.setitem(sys.modules, "torch", None)

    # Act
    result = check_gpu()

    # Assert
    assert result.status == HealthStatus.NOT_CONFIGURED
    assert "PyTorch" in result.description


def test_check_plutus_when_model_loads_should_return_healthy_with_model_data():
    # Arrange
    generator = MagicMock(sequence_length=30)

    with patch(FORECAST_GENERATOR, return_value=generator):
        # Act
        result = check_plutus()

    # Assert
    assert result.status == HealthStatus.HEALTHY
    assert result.data == {"model_name": "plutus-forecasting-v1", "sequence_length": 30}


def test_check_plutus_when_artifacts_missing_should_return_not_configured():
    # Arrange
    missing = FileNotFoundError("models/plutus-forecasting-v1/params.json")

    with patch(FORECAST_GENERATOR, side_effect=missing):
        # Act
        result = check_plutus()

    # Assert
    assert result.status == HealthStatus.NOT_CONFIGURED
    assert "not found" in result.description


def test_check_plutus_when_load_fails_should_return_degraded_not_unhealthy():
    # Arrange
    with patch(FORECAST_GENERATOR, side_effect=KeyError("hidden_size")):
        # Act
        result = check_plutus()

    # Assert
    assert result.status == HealthStatus.DEGRADED
    assert "failed to load" in result.description


def test_check_plutus_when_load_fails_twice_should_warn_once_without_traceback():
    # Arrange
    with patch(FORECAST_GENERATOR, side_effect=KeyError("hidden_size")), capture_logs() as logs:
        # Act
        check_plutus()
        check_plutus()

    # Assert
    assert logs == [
        {
            "event": "health check failed",
            "log_level": "warning",
            "check": "plutus",
            "error_type": "KeyError",
            "error": "'hidden_size'",
        }
    ]


@pytest.mark.asyncio
async def test_in_thread_when_awaited_should_run_sync_check_off_the_event_loop():
    # Arrange
    loop_thread = threading.get_ident()
    ran_on: list[int] = []

    def sync_check():
        ran_on.append(threading.get_ident())
        return HealthCheck(status=HealthStatus.HEALTHY, description="ok")

    # Act
    result = await service._in_thread(sync_check)(5.0)

    # Assert
    assert result.status == HealthStatus.HEALTHY
    assert ran_on and ran_on[0] != loop_thread


@pytest.mark.asyncio
async def test_handle_when_all_checks_healthy_should_return_healthy():
    # Arrange
    async def healthy_check(*args, **kwargs):
        return HealthCheck(status=HealthStatus.HEALTHY, description="ok")

    checks = {
        "test_a": healthy_check,
        "test_b": healthy_check,
    }

    with patch("ouranos_ml.features.health.check.service._CHECKS", checks):
        # Act
        result = await handle()

    # Assert
    assert result.status == HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_handle_when_one_check_unhealthy_should_return_unhealthy():
    # Arrange
    async def healthy_check(*args, **kwargs):
        return HealthCheck(status=HealthStatus.HEALTHY, description="ok")

    async def unhealthy_check(*args, **kwargs):
        return HealthCheck(status=HealthStatus.UNHEALTHY, description="bad")

    checks = {
        "test_a": healthy_check,
        "test_b": unhealthy_check,
    }

    with patch("ouranos_ml.features.health.check.service._CHECKS", checks):
        # Act
        result = await handle()

    # Assert
    assert result.status == HealthStatus.UNHEALTHY
    assert result.checks["test_b"].status == HealthStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_handle_when_one_check_degraded_none_unhealthy_should_return_degraded():
    # Arrange
    async def healthy_check(*args, **kwargs):
        return HealthCheck(status=HealthStatus.HEALTHY, description="ok")

    async def degraded_check(*args, **kwargs):
        return HealthCheck(status=HealthStatus.DEGRADED, description="degraded")

    checks = {
        "test_a": healthy_check,
        "test_b": degraded_check,
    }

    with patch("ouranos_ml.features.health.check.service._CHECKS", checks):
        # Act
        result = await handle()

    # Assert
    assert result.status == HealthStatus.DEGRADED


@pytest.mark.asyncio
async def test_handle_when_all_checks_not_configured_should_return_not_configured():
    # Arrange
    async def not_configured_check(*args, **kwargs):
        return HealthCheck(status=HealthStatus.NOT_CONFIGURED, description="not configured")

    checks = {
        "test_a": not_configured_check,
        "test_b": not_configured_check,
    }

    with patch("ouranos_ml.features.health.check.service._CHECKS", checks):
        # Act
        result = await handle()

    # Assert
    assert result.status == HealthStatus.NOT_CONFIGURED


@pytest.mark.asyncio
async def test_handle_when_check_raises_exception_should_return_unhealthy():
    # Arrange
    async def healthy_check(*args, **kwargs):
        return HealthCheck(status=HealthStatus.HEALTHY, description="ok")

    async def raising_check(*args, **kwargs):
        raise RuntimeError("boom")

    checks = {
        "test_a": healthy_check,
        "test_b": raising_check,
    }

    with patch("ouranos_ml.features.health.check.service._CHECKS", checks):
        # Act
        result = await handle()

    # Assert
    assert result.status == HealthStatus.UNHEALTHY
    assert result.checks["test_b"].status == HealthStatus.UNHEALTHY
    assert result.checks["test_b"].description == "boom"
