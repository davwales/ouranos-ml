from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class SequenceConfig:
    """Configuration for a sequence-based data processing pipeline."""

    sequence_length: int
    prediction_horizon: int
    feature_columns: list[str]
    target_columns: list[str]
    group_by_column: str | None = None
    sort_by_column: str | None = None
    sequence_transform: Callable[[np.ndarray], np.ndarray] | None = None
    target_transform: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None


class SequenceProcessor:
    """Sequence processor for preparing data for sequence-based models."""

    def __init__(self, config: SequenceConfig) -> None:
        self.config = config

    def _validate_data(self, df: pl.DataFrame) -> None:
        required_columns = set(self.config.feature_columns) | set(self.config.target_columns)
        if self.config.group_by_column:
            required_columns.add(self.config.group_by_column)
        if self.config.sort_by_column:
            required_columns.add(self.config.sort_by_column)

        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _process_group(self, group_data: pl.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
        feature_data = group_data.select(self.config.feature_columns).to_numpy()
        target_data = group_data.select(self.config.target_columns).to_numpy()

        sequences = []
        targets = []

        for i in range(len(feature_data) - self.config.sequence_length - self.config.prediction_horizon + 1):
            sequence = feature_data[i : (i + self.config.sequence_length)]

            target = target_data[
                i + self.config.sequence_length : i + self.config.sequence_length + self.config.prediction_horizon
            ]

            if self.config.target_transform:
                target = self.config.target_transform(target, sequence)

            if self.config.sequence_transform:
                sequence = self.config.sequence_transform(sequence)

            sequences.append(sequence)
            targets.append(target)

        return sequences, targets

    def create_sequences(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Creates sequences from the given dataframe."""
        self._validate_data(df)

        # Groups come out in key order; rows keep their input order unless a sort column is set.
        sort_columns = [column for column in (self.config.group_by_column, self.config.sort_by_column) if column]
        if sort_columns:
            df = df.sort(sort_columns, maintain_order=True)
        groups = (
            df.partition_by(self.config.group_by_column, maintain_order=True) if self.config.group_by_column else [df]
        )

        all_sequences = []
        all_targets = []
        for group_data in groups:
            sequences, targets = self._process_group(group_data)
            all_sequences.extend(sequences)
            all_targets.extend(targets)

        return np.array(all_sequences), np.array(all_targets)
