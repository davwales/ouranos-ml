import logging
import sys

from experiments.base_experiment import BaseExperiment
from experiments.plutus_forecasting.experiment import PlutusForecastingExperiment

logging.basicConfig(level=logging.DEBUG)


def main() -> None:
    """The main entry point to run experiments."""
    if len(sys.argv) < 2:
        raise ValueError("Please provide the name of the experiment(s) you wish to run")

    experiments: list[BaseExperiment] = [PlutusForecastingExperiment()]

    target_experiment_names = sys.argv[1:]
    for target_experiment_name in target_experiment_names:
        try:
            target_experiment: BaseExperiment | None = next(
                (e for e in experiments if e.NAME == target_experiment_name), None
            )

            if target_experiment is None:
                logging.warning(f"Failed to find experiment '{target_experiment_name}'.")
                continue

            target_experiment.run()
        except Exception as e:
            logging.error(f"Failed to run experiment '{target_experiment_name}' with error '{str(e)}'")
