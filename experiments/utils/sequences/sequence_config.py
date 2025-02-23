from dataclasses import dataclass
from typing import List, Callable, Optional
import numpy as np

@dataclass
class SequenceConfig:
    sequence_length: int
    prediction_horizon: int
    feature_columns: List[str]
    target_columns: List[str]
    group_by_column: Optional[str] = None
    sort_by_column: Optional[str] = None
    sequence_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    target_transform: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None