from ouranos_ml.shared.domain.core.base_schema import BaseSchema


class ChildSchema(BaseSchema):
    """Test child schema for nesting tests."""

    child_name: str
    child_age: int


class ParentSchema(BaseSchema):
    """Test parent schema containing a child."""

    parent_name: str
    child: ChildSchema


class SimpleSchema(BaseSchema):
    """Test schema with basic fields."""

    user_name: str
    is_active: bool


class SimpleObj:
    """Test object with attributes for from_attributes tests."""

    def __init__(self, user_name: str, is_active: bool) -> None:
        self.user_name = user_name
        self.is_active = is_active


def test_base_schema_when_snake_case_fields_should_serialize_to_camel_case() -> None:
    """Test that BaseSchema serializes snake_case to camelCase."""
    # Arrange
    instance = SimpleSchema(user_name="alice", is_active=True)

    # Act
    dumped = instance.model_dump(by_alias=True)

    # Assert
    assert "userName" in dumped
    assert "isActive" in dumped
    assert dumped["userName"] == "alice"
    assert dumped["isActive"] is True


def test_base_schema_when_camel_case_input_should_deserialize_to_snake_case() -> None:
    """Test that BaseSchema deserializes camelCase input to snake_case fields."""
    # Arrange

    # Act
    instance = SimpleSchema.model_validate({"userName": "bob", "isActive": False})

    # Assert
    assert instance.user_name == "bob"
    assert instance.is_active is False


def test_base_schema_when_populate_by_name_should_accept_both_casings() -> None:
    """Test that BaseSchema accepts both snake_case and camelCase input."""
    # Arrange

    # Act
    snake_instance = SimpleSchema.model_validate({"user_name": "carol", "is_active": True})
    camel_instance = SimpleSchema.model_validate({"userName": "carol", "isActive": True})

    # Assert
    assert snake_instance.user_name == camel_instance.user_name
    assert snake_instance.is_active == camel_instance.is_active


def test_base_schema_when_nested_models_should_round_trip_correctly() -> None:
    """Test that nested BaseSchema models round-trip through serialization."""
    # Arrange
    original = ParentSchema(parent_name="dad", child=ChildSchema(child_name="kid", child_age=5))

    # Act
    dumped = original.model_dump(by_alias=True)
    restored = ParentSchema.model_validate(dumped)

    # Assert
    assert restored.parent_name == "dad"
    assert restored.child.child_name == "kid"
    assert restored.child.child_age == 5


def test_base_schema_when_from_attributes_true_should_build_from_object() -> None:
    """Test that BaseSchema can be built from a plain object via from_attributes."""
    # Arrange
    obj = SimpleObj(user_name="dave", is_active=True)

    # Act
    instance = SimpleSchema.model_validate(obj)

    # Assert
    assert instance.user_name == "dave"
    assert instance.is_active is True
