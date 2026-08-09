from ouranos_ml.features.health.check.schemas import (
    CheckStatus,
    GpuCheckData,
    HealthCheck,
    HealthResponse,
    ServiceStatus,
)


def test_check_status_when_healthy_should_serialize_to_healthy_string():
    # Arrange
    status = CheckStatus.HEALTHY

    # Act
    value = status.value

    # Assert
    assert value == "healthy"


def test_health_check_when_created_should_set_timestamp():
    # Arrange
    from datetime import UTC, datetime

    before = datetime.now(tz=UTC)

    # Act
    check = HealthCheck(status=CheckStatus.HEALTHY, description="ok")

    # Assert
    after = datetime.now(tz=UTC)
    assert before <= check.timestamp <= after


def test_health_check_when_data_provided_should_serialize_data():
    # Arrange
    check = HealthCheck(status=CheckStatus.HEALTHY, description="ok", data={"device_count": 1})

    # Act
    result = check.model_dump(by_alias=True)

    # Assert
    assert result["data"] == {"device_count": 1}


def test_gpu_check_data_when_serialized_should_use_camel_case():
    # Arrange
    gpu_data = GpuCheckData(device_count=2)

    # Act
    result = gpu_data.model_dump(by_alias=True)

    # Assert
    assert result == {"deviceCount": 2}


def test_health_response_when_serialized_should_include_checks_dict():
    # Arrange
    check = HealthCheck(status=CheckStatus.HEALTHY, description="LLM backend is reachable")
    response = HealthResponse(status=ServiceStatus.HEALTHY, checks={"llm": check})

    # Act
    result = response.model_dump(by_alias=True)

    # Assert
    assert result["status"] == "healthy"
    assert "llm" in result["checks"]
    assert result["checks"]["llm"]["status"] == "healthy"
