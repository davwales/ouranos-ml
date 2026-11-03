from abc import ABC, abstractmethod
from argparse import ArgumentParser


class BaseExperiment(ABC):
    """Base class for all experiments."""

    NAME: str

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        """Adds the experiment's CLI options. Parsed values are passed to ``__init__`` as keyword arguments."""

    @abstractmethod
    def run(self) -> None:
        """Main entry point for the experiment."""
