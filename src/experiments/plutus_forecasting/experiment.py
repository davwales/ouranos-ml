import logging
from typing import Any
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from experiments.base_experiment import BaseExperiment
from experiments.plutus_forecasting.model import Model
from experiments.plutus_forecasting.time_series_dataset import TimeSeriesDataset
from experiments.utils.data_splitters import split_by_bucket
from experiments.utils.genetic_tuner import GeneticTuner
from experiments.utils.harness import TrainingHarness
from experiments.utils.sequences import SequenceConfig, SequenceProcessor


class PlutusForecastingExperiment(BaseExperiment):
    """Experiment attempting to forecast financial markets using plutus data."""

    NAME: str = "plutus_forecasting"

    def run(self) -> None:
        """Main entry point for the plutus forecasting experiment."""
        logging.info("Starting the plutus forecasting experiment.")

        logging.info("Loading data...")
        df = pd.read_csv("datasets/plutus.trades.training.daily.csv")

        # It can be helpful during development to limit the number of symbols being processed.
        unique_symbols = df["symbolId"].unique()
        df = df[df["symbolId"].isin(unique_symbols[:100])]

        logging.info("Splitting data...")
        train_df, val_df, test_df = split_by_bucket(df, bucket_field="bucket")

        base_path = "src/experiments/plutus_forecasting"
        features = ["averagePrice", "minPrice", "maxPrice", "volume"]
        prediction_horizon = 1
        param_space = {
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
            sequence_transform=lambda seq: seq / np.max(seq, axis=0),
            target_transform=lambda target, seq: np.clip(target / np.max(seq, axis=0), 0.0, 1.5),
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
            file=f"{base_path}/params.json",
        )

        logging.info(f"Using params: {params}")
        harness = self._create_harness(x_train.shape[2], prediction_horizon, y_train.shape[2], params)
        harness.train(train_loader, val_loader, epochs=1000, early_stopping=10)

        logging.info("Validating model...")
        test_loss = harness.validate(test_loader)
        logging.info(f"Test loss: {test_loss:.4f}")

        logging.info("Saving model...")
        harness.save_model(f"{base_path}/model_{uuid4()}.pth")

        predictions = harness.predict(torch.FloatTensor(x_test))
        self._plot_predictions(y_test, predictions, features)
        plt.savefig(f"{base_path}/predictions.png")
        plt.show()

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
    ) -> None:
        n_features = y_test.shape[-1]
        if feature_names is None:
            feature_names = [f"Feature {i + 1}" for i in range(n_features)]

        fig, axes = plt.subplots(1, n_features, figsize=(15, 5))
        if n_features == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            actual = y_test[:, :, i]
            pred = predictions[:, :, i]
            mask = np.random.choice(actual.shape[0], sample, replace=False)

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

        plt.tight_layout()
