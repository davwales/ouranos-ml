import numpy as np
import torch


class TimeSeriesDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch time series dataset."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x: torch.FloatTensor = torch.FloatTensor(x)
        self.y: torch.FloatTensor = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get item from the dataset."""
        return self.x[index], self.y[index]
