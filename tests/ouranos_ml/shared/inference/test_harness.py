"""Tests for ouranos_ml.shared.inference.harness."""

import pytest

from tests.conftest import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="real torch is not installed")

if TORCH_AVAILABLE:
    import numpy as np
    import torch

    from ouranos_ml.shared.inference.harness import Harness, TrainingHarness
    from ouranos_ml.shared.inference.model import Model

_CPU = torch.device("cpu")


def _tiny_model() -> "Model":
    """Build a tiny CPU forecasting model."""
    return Model(input_size=2, hidden_size=4, prediction_horizon=1, output_size=2, num_layers=1, dropout=0.0)


@pytest.fixture
def harness() -> "Harness":
    """Build a harness around a tiny CPU model."""
    return Harness(_tiny_model(), device=_CPU)


@pytest.fixture
def training_harness() -> "TrainingHarness":
    """Build a training harness with Adam and L1 loss on a tiny CPU model."""
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return TrainingHarness(model, optimizer, torch.nn.L1Loss(), device=_CPU)


def test_harness_when_predict_called_should_return_numpy_array(harness):
    # Arrange
    batch = torch.randn(4, 5, 2)

    # Act
    output = harness.predict(batch)

    # Assert
    assert isinstance(output, np.ndarray)
    assert output.shape == (4, 1, 2)


def test_harness_when_load_model_with_missing_path_should_raise(harness, tmp_path):
    # Arrange
    missing = str(tmp_path / "missing.pth")

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        harness.load_model(missing)


def test_harness_when_save_then_load_should_round_trip(training_harness, tmp_path):
    # Arrange
    batch = torch.randn(4, 5, 2)
    path = str(tmp_path / "model.pth")

    # Act
    training_harness.save_model(path)
    restored = Harness(_tiny_model(), device=_CPU)
    restored.load_model(path)

    # Assert
    assert np.allclose(training_harness.predict(batch), restored.predict(batch))


def test_training_harness_when_train_called_should_restore_best_epoch_weights(training_harness):
    # Arrange
    weight_delta = 0.1
    scripted_train_losses = iter([1.0, 0.5, 0.6, 0.7])
    scripted_val_losses = iter([0.9, 0.4, 0.5, 0.6])
    epochs_run = []

    original_state = {key: value.clone() for key, value in training_harness.model.state_dict().items()}

    def fake_train_epoch(loader):
        with torch.no_grad():
            for parameter in training_harness.model.parameters():
                parameter.add_(weight_delta)
        epochs_run.append(len(epochs_run))
        return next(scripted_train_losses)

    def fake_validate(loader):
        return next(scripted_val_losses)

    training_harness._train_epoch = fake_train_epoch
    training_harness.validate = fake_validate

    # Act
    training_harness.train(train_loader=None, val_loader=None, epochs=4, early_stopping=3)

    # Assert
    # Epoch 1 had the lowest val loss (0.4); no early stop fired, so the final
    # epoch's weights would be original + delta * 4 without the best-model restore.
    best_epoch = 1
    for key, value in training_harness.model.state_dict().items():
        expected = original_state[key] + weight_delta * (best_epoch + 1)
        assert torch.allclose(value, expected), f"weights for '{key}' were not restored from the best epoch"
    assert len(epochs_run) == 4


def test_training_harness_when_val_loss_never_improves_should_stop_after_patience(training_harness):
    # Arrange
    train_losses = iter([0.5, 0.5, 0.5])
    val_losses = iter([1.0, 1.0, 1.0])
    epochs_run = []

    def fake_train_epoch(loader):
        epochs_run.append(len(epochs_run))
        return next(train_losses)

    def fake_validate(loader):
        return next(val_losses)

    training_harness._train_epoch = fake_train_epoch
    training_harness.validate = fake_validate

    # Act
    training_harness.train(train_loader=None, val_loader=None, epochs=3, early_stopping=2)

    # Assert
    assert len(epochs_run) == 3


def test_training_harness_when_zero_epochs_should_raise(training_harness):
    # Act & Assert
    with pytest.raises(ValueError, match="Failed to train"):
        training_harness.train(train_loader=None, val_loader=None, epochs=0)