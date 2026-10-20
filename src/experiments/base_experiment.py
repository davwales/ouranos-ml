from abc import ABC, abstractmethod


class BaseExperiment(ABC):
    """Base class for all experiments."""

    NAME: str

    @abstractmethod
    def run(self) -> None:
        """Main entry point for the experiment."""
