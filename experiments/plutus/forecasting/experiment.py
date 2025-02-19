import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from .model import Model
from .time_series_dataset import TimeSeriesDataset
from experiments.harness import Harness

def create_train_val_test_split(df, val_ratio=0.1, test_ratio=0.1):
    unique_buckets = sorted(df['bucket'].unique())
    n_buckets = len(unique_buckets)
    
    # Calculate split indices
    test_idx = int(n_buckets * (1 - test_ratio))
    val_idx = int(n_buckets * (1 - test_ratio - val_ratio))
    
    # Split buckets
    train_buckets = unique_buckets[:val_idx]
    val_buckets = unique_buckets[val_idx:test_idx]
    test_buckets = unique_buckets[test_idx:]
    
    # Create masks
    train_mask = df['bucket'].isin(train_buckets)
    val_mask = df['bucket'].isin(val_buckets)
    test_mask = df['bucket'].isin(test_buckets)
    
    return df[train_mask], df[val_mask], df[test_mask]

def prepare_samples(df, sequence_length=10, prediction_horizon=1):
    df = df.sort_values(['bucket', 'symbolId'])
    unique_symbols = df['symbolId'].unique()

    symbol_scalers = {}
    sequences = []
    targets = []
    
    for symbol in unique_symbols:
        symbol_data = df[df['symbolId'] == symbol].sort_values('bucket')
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        price_scaler = scaler.fit(symbol_data[['minPrice', 'maxPrice', 'averagePrice']])
        
        # Scale volume separately (using log transformation for better distribution)
        volume_scaler = MinMaxScaler(feature_range=(0, 1))
        symbol_data['scaled_volume'] = volume_scaler.fit_transform(
            np.log1p(symbol_data[['volume']])
        )
        
        # Scale prices
        scaled_prices = price_scaler.transform(
            symbol_data[['minPrice', 'maxPrice', 'averagePrice']]
        )
        
        # Combine scaled features
        scaled_data = np.hstack((
            scaled_prices,
            symbol_data['scaled_volume'].values.reshape(-1, 1)
        ))
        
        # Store scalers
        symbol_scalers[symbol] = {
            'price': price_scaler,
            'volume': volume_scaler
        }

        # Create sequences for this symbol
        for i in range(len(scaled_data) - sequence_length - prediction_horizon + 1):
            sequence = scaled_data[i:(i + sequence_length)]
            sequences.append(sequence)
            
            # Use average price as target
            target = symbol_data.iloc[i + sequence_length + prediction_horizon - 1]['averagePrice']
            targets.append(target)
    
    return np.array(sequences), np.array(targets), symbol_scalers

if __name__ == "__main__":
    # Expected training data columns:
    # symbolId,bucket,minPrice,maxPrice,averagePrice,volume

    print("Loading data...")
    df = pd.read_csv('datasets/plutus.trades.training.csv')
    
    # It can be helpful during development to limit the number of symbols being processed.
    unique_symbols = df['symbolId'].unique()
    df = df[df['symbolId'].isin(unique_symbols[:100])]

    print("Splitting data...")
    train_df, val_df, test_df = create_train_val_test_split(df)

    x_train, y_train, scalers = prepare_samples(train_df, sequence_length=10, prediction_horizon=1)
    train_dataset = TimeSeriesDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    print(f'Training data shape: {x_train.shape}, {y_train.shape}')

    x_val, y_val, _ = prepare_samples(val_df, sequence_length=10, prediction_horizon=1)
    val_dataset = TimeSeriesDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    print(f'Validation data shape: {x_val.shape}, {y_val.shape}')

    x_test, y_test, _ = prepare_samples(test_df, sequence_length=10, prediction_horizon=1)
    test_dataset = TimeSeriesDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f'Test data shape: {x_test.shape}, {y_test.shape}')
    
    print("Training model...")
    model = Model(x_train.shape[2], hidden_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    harness = Harness(model, optimizer, loss_fn)
    harness.train(train_loader, val_loader, epochs=10)

    print("Testing model...")
    test_loss = harness.validate(test_loader)
    print(f'Test Loss: {test_loss:.4f}')

    for i in range(min(10, len(x_test))):
        prediction = harness.predict(torch.FloatTensor(x_test[i]).unsqueeze(0))[0][0]
        print(f'Predicted: {prediction}, Actual: {y_test[i]}')