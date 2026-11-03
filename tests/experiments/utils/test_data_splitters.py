"""Tests for experiments.utils.data_splitters."""

import pytest

pl = pytest.importorskip("polars", reason="the experiments dependency group is not installed")

from experiments.utils.data_splitters import split_by_bucket  # noqa: E402


def test_split_by_bucket_when_rows_unsorted_should_split_whole_buckets_in_bucket_order():
    # Arrange
    buckets = [f"2025-01-{day:02d}" for day in range(10, 0, -1)]
    df = pl.DataFrame({"bucket": buckets * 2, "symbol": ["a"] * 10 + ["b"] * 10})

    # Act
    train, val, test = split_by_bucket(df, bucket_field="bucket")

    # Assert
    assert sorted(set(train["bucket"])) == [f"2025-01-{day:02d}" for day in range(1, 9)]
    assert set(val["bucket"]) == {"2025-01-09"}
    assert set(test["bucket"]) == {"2025-01-10"}
    assert (len(train), len(val), len(test)) == (16, 2, 2)


def test_split_by_bucket_when_split_should_keep_input_row_order():
    # Arrange
    df = pl.DataFrame({"bucket": [3, 1, 2, 1, 3], "row": [0, 1, 2, 3, 4]})

    # Act
    train, _, _ = split_by_bucket(df, bucket_field="bucket", val_ratio=0.0, test_ratio=0.0)

    # Assert
    assert train["row"].to_list() == [0, 1, 2, 3, 4]
