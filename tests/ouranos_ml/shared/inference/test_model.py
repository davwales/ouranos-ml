"""Tests for ouranos_ml.shared.inference.model."""

import pytest

from tests.conftest import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="real torch is not installed")

if TORCH_AVAILABLE:
    import torch

    from ouranos_ml.shared.inference.model import Model


@pytest.fixture
def model() -> "Model":
    """Build a tiny CPU model for shape tests."""
    return Model(
        input_size=2,
        hidden_size=4,
        prediction_horizon=1,
        output_size=2,
        num_layers=1,
        dropout=0.0,
    )


def test_model_when_forward_called_should_return_expected_shape(model):
    # Arrange
    batch = torch.randn(4, 5, 2)

    # Act
    output = model(batch)

    # Assert
    assert output.shape == (4, 1, 2)


def test_model_when_forward_called_on_single_sequence_should_return_expected_shape(model):
    # Arrange
    batch = torch.randn(1, 30, 2)

    # Act
    output = model(batch)

    # Assert
    assert output.shape == (1, 1, 2)