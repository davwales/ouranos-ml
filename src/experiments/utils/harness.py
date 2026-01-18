import logging
import os
from collections.abc import Callable
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader


class _StepScheduler(Protocol):
    """Protocol used to define schedulers that can be used with the harness."""

    def step(self, loss: int | float) -> None: ...


class Harness:
    """Harness meant to simplify the management of ML model usage."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logging.debug(f"Initialized harness with device '{self.device}'")

    def predict(self, input: torch.Tensor) -> np.ndarray:
        """Predicts the next value given an input sequence."""
        self.model.eval()
        with torch.no_grad():
            input = input.to(self.device)
            return self.model(input).cpu().numpy()

    def load_model(self, path: str) -> None:
        """Loads a saved model from the given file."""
        if not os.path.exists(path):
            logging.error(f"No model found at path '{path}'.")
            return
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

    def save_model(self, path: str) -> None:
        """Saves the current model to the given path."""
        torch.save(self.model.state_dict(), path)


class TrainingHarness(Harness):
    """Harness with support for training a model."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: _StepScheduler | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(model, device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

    def validate(self, val_loader: DataLoader) -> float:
        """Validates the current model against the validation dataset."""
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()
        return val_loss / len(val_loader)

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, early_stopping: int | None = None
    ) -> None:
        """Trains the model for a specified number of epochs"""
        best_val_loss = float("inf")
        best_model = None
        epochs_no_improve = 0
        for _epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            epochs_no_improve += 1
            additional_message = ""
            if self.scheduler:
                self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                epochs_no_improve = 0
                best_val_loss = val_loss
                best_model = self.model.state_dict().copy()
                additional_message = "<--- New Best"
            if early_stopping and epochs_no_improve >= early_stopping:
                logging.debug(f"Early stopping at epoch {_epoch}")
                break
            logging.debug(f"Epoch: {_epoch} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} {additional_message}")

        if best_model is None:
            raise ValueError("Failed to train a model.")

        self.model.load_state_dict(best_model)

    def _train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.loss_fn(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
