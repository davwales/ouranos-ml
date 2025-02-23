import random
import json
import os
from copy import deepcopy
from typing import List, Dict, Callable, Optional

from experiments.harness import Harness

class GeneticTuner:
    def __init__(
        self,
        param_space: Dict[str, List[float]],
        population_size: int = 20,
        elite_size: int = 2,
        mutation_rate: float = 0.1,
        harness_creator: Callable[[Dict[str, float]], Harness] = None
    ):
        self.param_space = param_space
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.harness_creator = harness_creator
        
    def _create_individual(self) -> Dict[str, float]:
        """Create a random individual from the parameter space"""
        return {
            key: random.choice(values)
            for key, values in self.param_space.items()
        }
    
    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Mutate an individual's parameters"""
        mutated = deepcopy(individual)
        for key in mutated:
            if random.random() < self.mutation_rate:
                mutated[key] = random.choice(self.param_space[key])
        return mutated
    
    def _crossover(
        self,
        parent1: Dict[str, float],
        parent2: Dict[str, float]
    ) -> Dict[str, float]:
        """Create a child from two parents using uniform crossover"""
        child = {}
        for key in parent1:
            child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        return child
    
    def evolve(
        self,
        generations: int,
        train_loader,
        val_loader,
        train_epochs: int = 10,
        early_stopping: Optional[int] = None,
        file: Optional[str] = None
    ):
        if file and os.path.exists(file):
            with open(file, 'r') as f:
                print('Loading params from file')
                return json.load(f)

        population = [self._create_individual() for _ in range(self.population_size)]
        best_fitness = float('inf')
        generations_no_improve = 0
        
        for generation in range(generations):
            fitness_scores = []
            for params in population:
                harness = self.harness_creator(params)
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
                print(f"Early stopping at generation {generation}")
                break
            
            sorted_population = [x for _, x in sorted(
                zip(fitness_scores, population),
                key=lambda pair: pair[0]
            )]
            
            # Select elite individuals
            new_population = sorted_population[:self.elite_size]
            
            # Create rest of new population
            while len(new_population) < self.population_size:
                parent1 = random.choice(sorted_population[:self.population_size//2])
                parent2 = random.choice(sorted_population[:self.population_size//2])
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
            print(f"Generation {generation + 1}/{generations}")
            print(f"Best fitness: {min(fitness_scores):.4f}")
            print(f"Best params: {sorted_population[0]}")
            print("-" * 50)
        
        if file:
            with open(file, 'w') as f:
                json.dump(sorted_population[0], f)
        return sorted_population[0]
