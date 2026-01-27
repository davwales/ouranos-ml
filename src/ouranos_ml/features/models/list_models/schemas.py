from ouranos_ml.shared.domain.core.base_schema import BaseSchema


class Model(BaseSchema):
    """Response schema for a single model."""

    id: str
    created: int
    object: str = "model"
    owned_by: str = "ouranos-ml"


class ListModelsResponse(BaseSchema):
    """Wrapper around the list of models being returned by the route."""

    object: str = "list"
    data: list[Model]
