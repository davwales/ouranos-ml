"""Factory functions for creating forecast test data."""

from ouranos_ml.features.plutus.forecast.schemas import ForecastBand, ForecastRequest
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint


def make_forecast_point(
    *,
    average_price: float = 10.0,
    min_price: float = 9.0,
    max_price: float = 11.0,
    volume: float = 1000.0,
) -> PlutusForecastPoint:
    """Create a PlutusForecastPoint with sensible defaults."""
    return PlutusForecastPoint(
        average_price=average_price,
        min_price=min_price,
        max_price=max_price,
        volume=volume,
    )


def make_sequence(length: int = 30) -> list[PlutusForecastPoint]:
    """Create a sequence of PlutusForecastPoint of the given length."""
    return [
        make_forecast_point(
            average_price=10.0 + i * 0.1,
            min_price=9.0 + i * 0.1,
            max_price=11.0 + i * 0.1,
            volume=1000.0 + i * 10.0,
        )
        for i in range(length)
    ]


def make_forecast_request(
    *,
    points: list[list[PlutusForecastPoint]] | None = None,
    num_predictions: int = 3,
    samples: int | None = None,
) -> ForecastRequest:
    """Create a ForecastRequest with sensible defaults."""
    return ForecastRequest(
        points=points if points is not None else [make_sequence()],
        num_predictions=num_predictions,
        samples=samples,
    )


def make_forecast_band(*, median_price: float = 10.0) -> ForecastBand:
    """Create a ForecastBand whose lower and upper points sit 10% either side of the median price."""
    return ForecastBand(
        lower=make_forecast_point(average_price=median_price * 0.9),
        median=make_forecast_point(average_price=median_price),
        upper=make_forecast_point(average_price=median_price * 1.1),
    )
