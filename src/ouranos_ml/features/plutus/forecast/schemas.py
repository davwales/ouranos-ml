from typing import Annotated

from ouranos_ml.shared.domain.core.base_schema import BaseSchema
from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint
from pydantic import Field, ValidationInfo, field_validator

# Pantheon batches up to 500 symbols per request (Plutus:Forecasting:BatchSize); keep headroom above that.
MAX_FORECAST_SEQUENCES = 1000
MAX_NUM_PREDICTIONS = 500
# Each sequence is forecast once per sample; bounding the total bounds a sampled request's GPU time.
MAX_SAMPLED_SEQUENCES = 10_000


class ForecastRequest(BaseSchema):
    """Request for predicting future Plutus datapoints."""

    points: list[Annotated[list[PlutusForecastPoint], Field(min_length=1)]] = Field(
        min_length=1, max_length=MAX_FORECAST_SEQUENCES
    )
    num_predictions: int = Field(ge=1, le=MAX_NUM_PREDICTIONS)
    samples: int | None = Field(
        default=None,
        ge=2,
        le=100,
        description="Monte Carlo dropout passes per sequence. When set, each prediction is a p10/p50/p90 band.",
    )

    @field_validator("samples")
    @classmethod
    def _check_sampled_sequences(cls, samples: int | None, info: ValidationInfo) -> int | None:
        # points is absent from info.data when it failed validation itself.
        points = info.data.get("points", [])
        if samples is not None and samples * len(points) > MAX_SAMPLED_SEQUENCES:
            raise ValueError(f"samples × number of sequences must be at most {MAX_SAMPLED_SEQUENCES}")
        return samples


class ForecastBand(BaseSchema):
    """A sampled prediction: the 10th, 50th and 90th percentiles of each field."""

    lower: PlutusForecastPoint
    median: PlutusForecastPoint
    upper: PlutusForecastPoint
