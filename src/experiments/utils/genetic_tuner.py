import json
import os
import random
from collections.abc import Callable
from copy import deepcopy

from torch.utils.data import DataLoader

from ouranos_ml.shared.inference.harness import TrainingHarness


class GeneticTuner:
    """Tuner used to tune hyperparameters for machine learning models using a genetic algorithm."""

    def __init__(
        self,
        param_space: dict[str, list[float | int]],
        harness_factory: Callable[[dict[str, float | int]], TrainingHarness],
        population_size: int = 20,
        elite_size: int = 2,
        mutation_rate: float = 0.1,
    ) -> None:
        self.param_space = param_space
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.harness_factory = harness_factory

    def _create_individual(self) -> dict[str, float | int]:
        """Create a random individual from the parameter space"""
        return {key: random.choice(values) for key, values in self.param_space.items()}

    def _mutate(self, individual: dict[str, float | int]) -> dict[str, float | int]:
        """Mutate an individual's parameters"""
        mutated = deepcopy(individual)
        for key in mutated:
            if random.random() < self.mutation_rate:
                mutated[key] = random.choice(self.param_space[key])
        return mutated

    def _crossover(self, parent1: dict[str, float | int], parent2: dict[str, float | int]) -> dict[str, float | int]:
        """Create a child from two parents using uniform crossover"""
        child = {}
        for key in parent1:
            child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        return child

    def evolve(
        self,
        generations: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_epochs: int = 10,
        early_stopping: int | None = None,
        file: str | None = None,
    ) -> dict[str, float | int]:
        """Evolves the hyperparameters by mutating and crossing over individuals in an effort to find the best
        performing values.

        If no early stopping is specified, the tuning will continue for the entire number of generations. If early
        stopping is set, the algorithm will stop after no improvement has been shown after the given number of
        generations.
        """
        if file and os.path.exists(file):
            with open(file) as f:
                return json.load(f)

        population = [self._create_individual() for _ in range(self.population_size)]
        best_fitness = float("inf")
        generations_no_improve = 0

        for _generation in range(generations):
            fitness_scores = []
            for params in population:
                harness = self.harness_factory(params)
                harness.train(train_loader, val_loader, train_epochs)
                val_loss = harness.validate(val_loader)
                fitness_scores.append(val_loss)

            # Check for early stopping
            current_best = min(fitness_scores)
            if current_best < best_fitness:
                best_fitness = current_best
                generations_no_improve = 0
            else:
                generations_no_improve += 1

            if early_stopping and generations_no_improve >= early_stopping:
                break

            sorted_population = [
                x for _, x in sorted(zip(fitness_scores, population, strict=False), key=lambda pair: pair[0])
            ]

            # Select elite individuals
            new_population = sorted_population[: self.elite_size]

            # Create rest of new population
            while len(new_population) < self.population_size:
                parent1 = random.choice(sorted_population[: self.population_size // 2])
                parent2 = random.choice(sorted_population[: self.population_size // 2])
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

        if file:
            with open(file, "w") as f:
                json.dump(sorted_population[0], f)
        return sorted_population[0]
