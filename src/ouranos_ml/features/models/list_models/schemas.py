from ouranos_ml.shared.domain.core.base_schema import BaseSchema


class ModelResponse(BaseSchema):
    """Response schema for a single model."""

    id: str
    created: int
    object: str = "model"
    owner: str = "ouranos-ml"


class ListModelsResponse(BaseSchema):
    """Wrapper around the list of models being returned by the route."""

    object: str = "list"
    data: list[ModelResponse]
