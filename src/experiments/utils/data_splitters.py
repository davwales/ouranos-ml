from typing import Any

import polars as pl


def split_by_bucket(
    df: pl.DataFrame, bucket_field: str, val_ratio: float = 0.1, test_ratio: float = 0.1
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Splits a DataFrame into training, validation, and testing sets.

    Datapoints are first grouped and sorted by the specified bucket field and then
    all datapoints within a specified bucket are assigned to a specific split.
    """
    unique_buckets: list[Any] = df[bucket_field].unique().sort().to_list()
    n_buckets = len(unique_buckets)

    # Calculate split indices
    test_idx = int(n_buckets * (1.0 - test_ratio))
    val_idx = int(n_buckets * (1.0 - test_ratio - val_ratio))

    # Split buckets
    train_buckets = unique_buckets[:val_idx]
    val_buckets = unique_buckets[val_idx:test_idx]
    test_buckets = unique_buckets[test_idx:]

    bucket = pl.col(bucket_field)
    return (
        df.filter(bucket.is_in(train_buckets)),
        df.filter(bucket.is_in(val_buckets)),
        df.filter(bucket.is_in(test_buckets)),
    )
