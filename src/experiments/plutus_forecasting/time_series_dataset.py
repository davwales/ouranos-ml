from typing import TypeVar

import torch

T = TypeVar("T")


class TimeSeriesDataset(torch.utils.data.Dataset[T]):
    """Torch time series dataset."""

    def __init__(self, x: T, y: T) -> None:
        self.x: torch.FloatTensor = torch.FloatTensor(x)
        self.y: torch.FloatTensor = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-method-override]
        """Get item from the dataset."""
        return self.x[idx], self.y[idx]
