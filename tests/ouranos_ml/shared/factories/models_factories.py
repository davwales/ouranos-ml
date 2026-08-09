"""Factory functions for creating model list test data."""

from ouranos_ml.features.models.list_models.schemas import (
    ListModelsResponse,
    Model,
)


def make_model(
    *,
    model_id: str = "test-model",
    created: int = 1700000000,
    owned_by: str = "ouranos-ml",
) -> Model:
    """Create a Model with sensible defaults."""
    return Model(id=model_id, created=created, owned_by=owned_by)


def make_list_models_response(
    *,
    models: list[Model] | None = None,
) -> ListModelsResponse:
    """Create a ListModelsResponse with sensible defaults."""
    return ListModelsResponse(
        data=models or [make_model(), make_model(model_id="test-model-2")],
    )
