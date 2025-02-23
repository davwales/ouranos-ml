from typing import List, Tuple
import numpy as np
import pandas as pd

from .sequence_config import SequenceConfig

class SequenceProcessor:
    def __init__(self, config: SequenceConfig):
        self.config = config
        
    def _validate_data(self, df: pd.DataFrame) -> None:
        required_columns = set(self.config.feature_columns) | set(self.config.target_columns)
        if self.config.group_by_column:
            required_columns.add(self.config.group_by_column)
        if self.config.sort_by_column:
            required_columns.add(self.config.sort_by_column)
            
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _process_group(self, group_data: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        feature_data = group_data[self.config.feature_columns].values
        target_data = group_data[self.config.target_columns].values
        
        sequences = []
        targets = []
            
        for i in range(len(feature_data) - self.config.sequence_length - self.config.prediction_horizon + 1):
            sequence = feature_data[i:(i + self.config.sequence_length)]
            
            target=target_data[
                i + self.config.sequence_length:
                i + self.config.sequence_length + self.config.prediction_horizon
            ]
            
            if self.config.target_transform:
                target = self.config.target_transform(target, sequence)

            if self.config.sequence_transform:
                sequence = self.config.sequence_transform(sequence)
                
            sequences.append(sequence)
            targets.append(target)
            
        return sequences, targets

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self._validate_data(df)
        
        all_sequences = []
        all_targets = []
        
        if self.config.group_by_column:
            if self.config.sort_by_column:
                df = df.sort_values([self.config.group_by_column, self.config.sort_by_column])
                
            for _, group_data in df.groupby(self.config.group_by_column):
                sequences, targets = self._process_group(group_data)
                all_sequences.extend(sequences)
                all_targets.extend(targets)
        else:
            if self.config.sort_by_column:
                df = df.sort_values(self.config.sort_by_column)
            sequences, targets = self._process_group(df)
            all_sequences.extend(sequences)
            all_targets.extend(targets)
            
        return np.array(all_sequences), np.array(all_targets)