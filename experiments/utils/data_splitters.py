def split(df, val_ratio=0.1, test_ratio=0.1):
    n = len(df)
    test_idx = int(n * (1.0 - test_ratio))
    val_idx = int(n * (1.0 - test_ratio - val_ratio))
    train_df = df[:val_idx]
    val_df = df[val_idx:test_idx]
    test_df = df[test_idx:]
    return train_df, val_df, test_df

def split_by_bucket(df, bucket_field, val_ratio=0.1, test_ratio=0.1):
    unique_buckets = sorted(df[bucket_field].unique())
    n_buckets = len(unique_buckets)
    
    # Calculate split indices
    test_idx = int(n_buckets * (1.0 - test_ratio))
    val_idx = int(n_buckets * (1.0 - test_ratio - val_ratio))
    
    # Split buckets
    train_buckets = unique_buckets[:val_idx]
    val_buckets = unique_buckets[val_idx:test_idx]
    test_buckets = unique_buckets[test_idx:]
    
    # Create masks
    train_mask = df[bucket_field].isin(train_buckets)
    val_mask = df[bucket_field].isin(val_buckets)
    test_mask = df[bucket_field].isin(test_buckets)
    
    return df[train_mask], df[val_mask], df[test_mask]

