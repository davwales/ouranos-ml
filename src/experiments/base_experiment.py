from abc import ABC


class BaseExperiment(ABC):
    """Base class for all experiments."""

    NAME: str

    def run(self) -> None:
        """Main entry point for the experiment."""
        raise ValueError("Experiment must implement the run method.")
