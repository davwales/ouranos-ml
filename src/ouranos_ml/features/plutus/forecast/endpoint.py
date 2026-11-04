from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError

from ouranos_ml.features.plutus.forecast.schemas import ForecastBand, ForecastRequest
from ouranos_ml.features.plutus.forecast.service import SequenceLengthError, forecast_bands, forecast_points
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint


def _forecast(request: ForecastRequest) -> list[list[PlutusForecastPoint]] | list[list[ForecastBand]]:
    """Predicts future datapoints given a sequence of prior Plutus datapoints.
    Predictions are made using the model trained from the 'plutus_forecasting' experiment. With ``samples`` set, each
    prediction is a p10/p50/p90 band from that many Monte Carlo dropout passes instead of a single point.

    Raises:
        RequestValidationError: A sequence does not match the model's sequence length (rendered as FastAPI's 422).
    """
    try:
        if request.samples is None:
            return forecast_points(request.points, request.num_predictions)
        return forecast_bands(request.points, request.num_predictions, request.samples)
    except SequenceLengthError as exc:
        # The expected length lives in the model's params.json, so it can only be checked once the model is loaded.
        raise RequestValidationError(
            [
                {
                    "type": "value_error",
                    "loc": ("body", "points", i),
                    "msg": f"Sequence must have exactly {exc.expected_length} points",
                    "input": len(request.points[i]),
                }
                for i in exc.invalid_indices
            ]
        ) from exc


def register(router: APIRouter) -> None:
    """Register Plutus forecasting endpoints on the provided router."""
    router.post("/forecast")(_forecast)
