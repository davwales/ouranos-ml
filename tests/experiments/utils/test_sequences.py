"""Tests for experiments.utils.sequences."""

import numpy as np
import pytest

pl = pytest.importorskip("polars", reason="the experiments dependency group is not installed")

from experiments.utils.sequences import SequenceConfig, SequenceProcessor  # noqa: E402


def _config(**overrides: object) -> SequenceConfig:
    """Build a two-point, one-step config over the value column."""
    defaults: dict = {
        "sequence_length": 2,
        "prediction_horizon": 1,
        "feature_columns": ["value"],
        "target_columns": ["value"],
    }
    return SequenceConfig(**{**defaults, **overrides})


def test_create_sequences_when_grouped_and_sorted_should_window_each_group_in_order():
    # Arrange
    df = pl.DataFrame(
        {
            "symbol": ["b", "a", "b", "a", "b", "a"],
            "day": [3, 2, 1, 1, 2, 3],
            "value": [23.0, 12.0, 21.0, 11.0, 22.0, 13.0],
        }
    )
    processor = SequenceProcessor(_config(group_by_column="symbol", sort_by_column="day"))

    # Act
    x, y = processor.create_sequences(df)

    # Assert
    np.testing.assert_array_equal(x[:, :, 0], [[11.0, 12.0], [21.0, 22.0]])
    np.testing.assert_array_equal(y[:, :, 0], [[13.0], [23.0]])


def test_create_sequences_when_grouped_without_sort_column_should_keep_row_order_within_groups():
    # Arrange
    df = pl.DataFrame({"symbol": ["b", "a", "b", "a", "b", "a"], "value": [3.0, 6.0, 2.0, 5.0, 1.0, 4.0]})
    processor = SequenceProcessor(_config(group_by_column="symbol"))

    # Act
    x, y = processor.create_sequences(df)

    # Assert
    np.testing.assert_array_equal(x[:, :, 0], [[6.0, 5.0], [3.0, 2.0]])
    np.testing.assert_array_equal(y[:, :, 0], [[4.0], [1.0]])


def test_create_sequences_when_transforms_set_should_scale_target_by_raw_sequence():
    # Arrange
    df = pl.DataFrame({"value": [1.0, 2.0, 4.0], "volume": [10, 20, 40]})
    processor = SequenceProcessor(
        _config(
            feature_columns=["value", "volume"],
            target_columns=["value", "volume"],
            sequence_transform=lambda seq: seq / seq.max(axis=0),
            target_transform=lambda target, seq: target / seq.max(axis=0),
        )
    )

    # Act
    x, y = processor.create_sequences(df)

    # Assert
    assert x.dtype == np.float64
    np.testing.assert_array_equal(x, [[[0.5, 0.5], [1.0, 1.0]]])
    np.testing.assert_array_equal(y, [[[2.0, 2.0]]])


def test_create_sequences_when_columns_missing_should_raise_value_error():
    # Arrange
    df = pl.DataFrame({"value": [1.0, 2.0, 3.0]})
    processor = SequenceProcessor(_config(group_by_column="symbol"))

    # Act & Assert
    with pytest.raises(ValueError, match="symbol"):
        processor.create_sequences(df)
