import argparse
import logging
import sys

from experiments.base_experiment import BaseExperiment
from experiments.plutus_forecasting.experiment import PlutusForecastingExperiment
from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.logging import configure_logging

EXPERIMENTS: list[type[BaseExperiment]] = [PlutusForecastingExperiment]


def main(argv: list[str] | None = None) -> None:
    """The main entry point to run experiments."""
    parser = argparse.ArgumentParser(prog="experiment", description="Run a training experiment.")
    subparsers = parser.add_subparsers(dest="experiment", required=True, metavar="EXPERIMENT")
    for experiment_cls in EXPERIMENTS:
        subparser = subparsers.add_parser(experiment_cls.NAME, help=experiment_cls.__doc__)
        experiment_cls.add_arguments(subparser)
        subparser.set_defaults(experiment_cls=experiment_cls)

    options = vars(parser.parse_args(argv))
    name = options.pop("experiment")
    experiment_cls = options.pop("experiment_cls")

    configure_logging(get_settings())
    try:
        experiment_cls(**options).run()
    except Exception:
        logging.exception(f"Failed to run experiment '{name}'")
        sys.exit(1)
