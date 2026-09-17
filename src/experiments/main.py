import logging
import sys

from experiments.base_experiment import BaseExperiment
from experiments.plutus_forecasting.experiment import PlutusForecastingExperiment
from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.logging import configure_logging


def main() -> None:
    """The main entry point to run experiments."""
    configure_logging(get_settings())

    if len(sys.argv) < 2:
        raise ValueError("Please provide the name of the experiment(s) you wish to run")

    experiments: list[BaseExperiment] = [PlutusForecastingExperiment()]

    target_experiment_names = sys.argv[1:]
    failed = False
    for target_experiment_name in target_experiment_names:
        try:
            target_experiment: BaseExperiment | None = next(
                (e for e in experiments if e.NAME == target_experiment_name), None
            )

            if target_experiment is None:
                logging.warning(f"Failed to find experiment '{target_experiment_name}'.")
                continue

            target_experiment.run()
        except Exception:
            logging.exception(f"Failed to run experiment '{target_experiment_name}'")
            failed = True

    if failed:
        sys.exit(1)
