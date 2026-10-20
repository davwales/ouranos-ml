"""Tests for experiments.utils.genetic_tuner."""

import json
import random

import pytest

from tests.conftest import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="real torch is not installed")

if TORCH_AVAILABLE:
    from experiments.utils.genetic_tuner import GeneticTuner

_PARAM_SPACE: dict[str, list[float | int]] = {"a": [1, 2, 3], "b": [10, 20, 30]}


def _fitness(params: dict[str, float | int]) -> float:
    """Deterministic validation loss: lower a and b are better."""
    return float(params["a"] + params["b"])


class _FakeHarness:
    """Harness stand-in that records training and scores params deterministically."""

    def __init__(self, params: dict[str, float | int], trained: list[dict[str, float | int]]) -> None:
        self.params = params
        self.trained = trained

    def train(self, train_loader: object, val_loader: object, epochs: int) -> None:
        """Record the training call."""
        self.trained.append(self.params)

    def validate(self, val_loader: object) -> float:
        """Return the deterministic fitness."""
        return _fitness(self.params)


@pytest.fixture
def trained() -> list[dict[str, float | int]]:
    """Every params set passed to train, in order."""
    return []


@pytest.fixture
def tuner(trained: list[dict[str, float | int]]) -> "GeneticTuner":
    """Build a small seeded tuner over the fake harness."""
    random.seed(0)
    return GeneticTuner(
        param_space=_PARAM_SPACE,
        harness_factory=lambda params: _FakeHarness(params, trained),
        population_size=6,
        elite_size=2,
    )


def test_evolve_when_output_directory_missing_should_create_it_and_write_best_params(tuner, trained, tmp_path):
    # Arrange
    file = tmp_path / "new-model" / "tuned_params.json"

    # Act
    best = tuner.evolve(generations=2, train_loader=None, val_loader=None, file=str(file))

    # Assert
    assert json.loads(file.read_text()) == best
    assert best == min(trained, key=_fitness)


def test_evolve_when_use_cached_and_file_exists_should_return_cached_param_space_keys(tuner, trained, tmp_path):
    # Arrange
    file = tmp_path / "tuned_params.json"
    file.write_text(json.dumps({"a": 3, "b": 30, "input_size": 4}))

    # Act
    best = tuner.evolve(generations=2, train_loader=None, val_loader=None, file=str(file), use_cached=True)

    # Assert
    assert best == {"a": 3, "b": 30}
    assert trained == []


def test_evolve_when_file_exists_without_use_cached_should_retune(tuner, trained, tmp_path):
    # Arrange
    file = tmp_path / "tuned_params.json"
    file.write_text(json.dumps({"a": 99, "b": 99}))

    # Act
    best = tuner.evolve(generations=2, train_loader=None, val_loader=None, file=str(file))

    # Assert
    assert trained != []
    assert best == min(trained, key=_fitness)
    assert json.loads(file.read_text()) == best


def test_evolve_when_cached_file_missing_param_keys_should_retune(tuner, trained, tmp_path):
    # Arrange
    file = tmp_path / "tuned_params.json"
    file.write_text(json.dumps({"a": 3, "input_size": 4}))

    # Act
    best = tuner.evolve(generations=1, train_loader=None, val_loader=None, file=str(file), use_cached=True)

    # Assert
    assert trained != []
    assert set(best) == set(_PARAM_SPACE)


def test_evolve_when_generations_zero_should_raise_value_error(tuner):
    # Act & Assert
    with pytest.raises(ValueError, match="generations"):
        tuner.evolve(generations=0, train_loader=None, val_loader=None)


def test_evolve_when_early_stopping_triggers_should_return_best_params_seen(tuner, trained):
    # Act
    best = tuner.evolve(generations=50, train_loader=None, val_loader=None, early_stopping=1)

    # Assert
    assert best == min(trained, key=_fitness)
