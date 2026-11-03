import argparse
import json
import logging
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from sklearn.metrics import r2_score

from experiments.base_experiment import BaseExperiment
from experiments.plutus_forecasting.time_series_dataset import TimeSeriesDataset
from experiments.utils.data_splitters import split_by_bucket
from experiments.utils.genetic_tuner import GeneticTuner
from experiments.utils.sequences import SequenceConfig, SequenceProcessor
from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.inference.harness import TrainingHarness
from ouranos_ml.shared.inference.model import Model

_DEFAULT_DATA_PATH = "datasets/plutus.trades.training.daily.csv"
_DEFAULT_SEED = 42
_DEFAULT_EPOCHS = 1000


class PlutusForecastingExperiment(BaseExperiment):
    """Experiment attempting to forecast financial markets using plutus data."""

    NAME: str = "plutus_forecasting"

    def __init__(
        self,
        data_path: str = _DEFAULT_DATA_PATH,
        output_dir: str | None = None,
        seed: int = _DEFAULT_SEED,
        symbol_limit: int | None = None,
        epochs: int = _DEFAULT_EPOCHS,
        retune: bool = True,
        show_plots: bool = False,
    ) -> None:
        """Initializes the experiment.

        Args:
            data_path: The daily trades CSV to train on.
            output_dir: Where the model artifact is written. None uses ``MODELS_DIR/PLUTUS_FORECAST_MODEL_NAME``,
                the directory the service loads from.
            seed: Seeds Python, NumPy and torch.
            symbol_limit: Train on only the first N symbols, useful for quick development runs. None trains on every
                symbol.
            epochs: The maximum number of epochs to train the final model; early stopping usually ends it sooner.
            retune: Run the genetic tuner. When False, the params from the last tuning run (``tuned_params.json``)
                are reused if present.
            show_plots: Display the prediction plot in a window after saving it. Blocks until the window is closed,
                so leave it off for headless runs.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.seed = seed
        self.symbol_limit = symbol_limit
        self.epochs = epochs
        self.retune = retune
        self.show_plots = show_plots

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Adds the experiment's CLI options."""
        parser.add_argument(
            "--data",
            dest="data_path",
            default=_DEFAULT_DATA_PATH,
            help="daily trades CSV to train on (default: %(default)s)",
        )
        parser.add_argument(
            "--output-dir", help="artifact directory (default: $MODELS_DIR/$PLUTUS_FORECAST_MODEL_NAME)"
        )
        parser.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="random seed (default: %(default)s)")
        parser.add_argument("--symbol-limit", type=int, help="train on only the first N symbols")
        parser.add_argument(
            "--epochs", type=int, default=_DEFAULT_EPOCHS, help="max epochs for the final model (default: %(default)s)"
        )
        parser.add_argument(
            "--retune",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="run the genetic tuner; --no-retune reuses tuned_params.json (default: %(default)s)",
        )
        parser.add_argument("--show-plots", action="store_true", help="show the prediction plot after saving it")

    def run(self) -> None:
        """Main entry point for the plutus forecasting experiment."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        logging.info("Starting the plutus forecasting experiment (seed=%s).", self.seed)

        logging.info("Loading data from %s...", self.data_path)
        df = pd.read_csv(self.data_path)

        if self.symbol_limit is not None:
            unique_symbols = df["symbolId"].unique()
            df = df[df["symbolId"].isin(unique_symbols[: self.symbol_limit])]
            logging.info("Limited data to the first %s of %s symbols.", self.symbol_limit, len(unique_symbols))

        logging.info("Splitting data...")
        train_df, val_df, test_df = split_by_bucket(df, bucket_field="bucket")

        settings = get_settings()
        base_path = self.output_dir or str(Path(settings.models_dir) / settings.plutus_forecast_model_name)
        Path(base_path).mkdir(parents=True, exist_ok=True)
        features = ["averagePrice", "minPrice", "maxPrice", "volume"]
        prediction_horizon = 1
        param_space: dict[str, list[float | int]] = {
            "learning_rate": [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
            "hidden_size": [32, 64, 128, 256, 512],
            "num_layers": [1, 2, 3],
            "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
        sequence_config = SequenceConfig(
            sequence_length=30,
            prediction_horizon=prediction_horizon,
            feature_columns=features,
            target_columns=features,
            group_by_column="symbolId",
            sort_by_column="bucket",
            sequence_transform=lambda seq: seq / np.maximum(np.max(seq, axis=0), 1e-8),
            target_transform=lambda target, seq: np.clip(target / np.maximum(np.max(seq, axis=0), 1e-8), 0.0, 1.5),
        )

        sequence_processor = SequenceProcessor(sequence_config)

        x_train, y_train = sequence_processor.create_sequences(train_df)
        train_dataset = TimeSeriesDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        logging.debug(f"Training data shape: {x_train.shape}, {y_train.shape}")

        x_val, y_val = sequence_processor.create_sequences(val_df)
        val_dataset = TimeSeriesDataset(x_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        logging.debug(f"Validation data shape: {x_val.shape}, {y_val.shape}")

        x_test, y_test = sequence_processor.create_sequences(test_df)
        test_dataset = TimeSeriesDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        logging.debug(f"Test data shape: {x_test.shape}, {y_test.shape}")

        logging.info("Training model...")
        tuner = GeneticTuner(
            population_size=10,
            param_space=param_space,
            harness_factory=lambda params: self._create_harness(
                x_train.shape[2], prediction_horizon, y_train.shape[2], params
            ),
        )

        params = tuner.evolve(
            generations=5,
            train_loader=train_loader,
            val_loader=val_loader,
            train_epochs=10,
            early_stopping=3,
            file=f"{base_path}/tuned_params.json",
            use_cached=not self.retune,
        )

        logging.info(f"Using params: {params}")
        harness = self._create_harness(x_train.shape[2], prediction_horizon, y_train.shape[2], params)
        harness.train(train_loader, val_loader, epochs=self.epochs, early_stopping=10)

        logging.info("Validating model...")
        test_loss = harness.validate(test_loader)
        baseline_loss = self._naive_baseline_loss(test_loader, harness.loss_fn)
        logging.info(f"Test loss: {test_loss:.4f} (last-value baseline: {baseline_loss:.4f})")

        logging.info("Saving model...")
        model_path = Path(base_path) / "model.pth"
        tmp_model_path = model_path.with_suffix(".pth.tmp")
        harness.save_model(str(tmp_model_path))
        os.replace(tmp_model_path, model_path)

        full_config: dict[str, Any] = {
            **params,
            "input_size": int(x_train.shape[2]),
            "output_size": int(y_train.shape[2]),
            "prediction_horizon": prediction_horizon,
            "sequence_length": 30,
            "feature_fields": ["average_price", "min_price", "max_price", "volume"],
        }

        with open(Path(base_path) / "params.json", "w") as f:
            json.dump(full_config, f, indent=4)
        logging.info("Saved model config: %s", full_config)

        if not self.show_plots:
            plt.switch_backend("Agg")
        predictions = harness.predict(torch.FloatTensor(x_test))
        fig = self._plot_predictions(y_test, predictions, features)
        fig.savefig(f"{base_path}/predictions.png")
        if self.show_plots:
            plt.show()
        plt.close(fig)

    def _naive_baseline_loss(
        self, loader: torch.utils.data.DataLoader, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> float:
        """Scores carrying each sequence's last value across the horizon, averaged like ``harness.validate``.

        Inputs and targets share the same per-sequence scaling, so the loss is directly comparable to the model's.
        """
        total = 0.0
        with torch.no_grad():
            for data, target in loader:
                prediction = data[:, -1:, :].expand(-1, target.shape[1], -1)
                total += loss_fn(prediction, target).item()
        return total / len(loader)

    def _create_harness(
        self, input_size: int, prediction_horizon: int, output_size: int, params: dict[str, Any]
    ) -> TrainingHarness:
        logging.info(f"Creating harness with params {params}")
        model = Model(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            prediction_horizon=prediction_horizon,
            output_size=output_size,
            num_layers=params["num_layers"],
            dropout=params["dropout"],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        loss_fn = torch.nn.L1Loss()
        return TrainingHarness(model, optimizer, loss_fn)

    def _plot_predictions(
        self, y_test: np.ndarray, predictions: np.ndarray, feature_names: list[str] | None = None, sample: int = 1000
    ) -> Figure:
        n_features = y_test.shape[-1]
        if feature_names is None:
            feature_names = [f"Feature {i + 1}" for i in range(n_features)]

        fig, axes = plt.subplots(1, n_features, figsize=(15, 5))
        if n_features == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            actual = y_test[:, :, i]
            pred = predictions[:, :, i]
            mask = np.random.choice(actual.shape[0], min(sample, actual.shape[0]), replace=False)

            ax.scatter(actual[mask], pred[mask], alpha=0.5)

            min_val = min(actual[mask].min(), pred[mask].min())
            max_val = max(actual[mask].max(), pred[mask].max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                label="Perfect Prediction",
            )

            r2 = r2_score(actual[mask], pred[mask])
            ax.text(
                0.05,
                0.95,
                f"R² = {r2:.3f}\n",
                transform=ax.transAxes,
                verticalalignment="top",
            )

            ax.set_title(f"{feature_names[i]}\nPredicted vs Actual")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.grid(True, alpha=0.3)

            if i == n_features - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        fig.tight_layout()
        return fig
