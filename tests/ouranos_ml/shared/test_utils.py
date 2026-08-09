from ouranos_ml.shared.utils import done_event, server_side_event
from pydantic import BaseModel


class SimpleModel(BaseModel):
    """Test model with a single field."""

    message: str


class NestedInner(BaseModel):
    """Test inner model for nesting tests."""

    inner_value: int


class NestedOuter(BaseModel):
    """Test outer model that nests another model."""

    outer_name: str
    inner: NestedInner


def test_server_side_event_when_no_event_name_provided_should_return_data_line_only() -> None:
    """Test SSE formatting without event name."""
    # Arrange
    model = SimpleModel(message="hello")

    # Act
    result = server_side_event(model)

    # Assert
    expected = f"data: {model.model_dump_json()}\n\n"
    assert result == expected
    assert "event:" not in result


def test_server_side_event_when_event_name_provided_should_include_event_line() -> None:
    """Test SSE formatting with event name."""
    # Arrange
    model = SimpleModel(message="hello")

    # Act
    result = server_side_event(model, event="update")

    # Assert
    expected = f"event: update\ndata: {model.model_dump_json()}\n\n"
    assert result == expected


def test_server_side_event_when_complex_model_should_dump_json_correctly() -> None:
    """Test SSE formatting with nested model."""
    # Arrange
    inner = NestedInner(inner_value=42)
    model = NestedOuter(outer_name="test", inner=inner)

    # Act
    result = server_side_event(model)

    # Assert
    expected = f"data: {model.model_dump_json()}\n\n"
    assert result == expected
    assert "outer_name" in model.model_dump_json() or "outerName" in model.model_dump_json()


def test_done_event_should_return_done_string() -> None:
    """Test done event returns correct SSE termination string."""
    # Arrange

    # Act
    result = done_event()

    # Assert
    assert result == "data: [DONE]\n\n"
