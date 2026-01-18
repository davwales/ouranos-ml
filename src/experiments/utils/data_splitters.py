from typing import Any

from pandas import DataFrame


def split(df: DataFrame, val_ratio: float = 0.1, test_ratio: float = 0.1) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Splits a DataFrame into training, validation, and testing sets"""
    n = len(df)
    test_idx = int(n * (1.0 - test_ratio))
    val_idx = int(n * (1.0 - test_ratio - val_ratio))
    return df[:val_idx], df[val_idx:test_idx], df[test_idx:]


def split_by_bucket(
    df: DataFrame, bucket_field: str, val_ratio: float = 0.1, test_ratio: float = 0.1
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Splits a DataFrame into training, validation, and testing sets.

    Datapoints are first grouped and sorted by the specified bucket field and then
    all datapoints within a specified bucket are assigned to a specific split.
    """
    unique_buckets: list[Any] = sorted(df[bucket_field].unique())
    n_buckets = len(unique_buckets)

    # Calculate split indices
    test_idx = int(n_buckets * (1.0 - test_ratio))
    val_idx = int(n_buckets * (1.0 - test_ratio - val_ratio))

    # Split buckets
    train_buckets = unique_buckets[:val_idx]
    val_buckets = unique_buckets[val_idx:test_idx]
    test_buckets = unique_buckets[test_idx:]

    # Create masks
    train_mask: DataFrame = df[bucket_field].isin(train_buckets)
    val_mask: DataFrame = df[bucket_field].isin(val_buckets)
    test_mask: DataFrame = df[bucket_field].isin(test_buckets)

    return df[train_mask], df[val_mask], df[test_mask]
