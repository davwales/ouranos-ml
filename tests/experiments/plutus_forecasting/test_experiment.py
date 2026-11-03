"""CPU smoke test for the plutus_forecasting experiment."""

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="real torch is not installed")

pl = pytest.importorskip("polars", reason="the experiments dependency group is not installed")

if TORCH_AVAILABLE:
    from experiments.main import main
    from ouranos_ml.features.plutus.forecast.service import ForecastGenerator
    from ouranos_ml.shared.domain.core.settings import get_settings
    from ouranos_ml.shared.domain.plutus.forecast_point import PlutusForecastPoint

_TUNED_PARAMS = {"learning_rate": 0.01, "hidden_size": 8, "num_layers": 1, "dropout": 0.0}


def _write_trades_csv(path: Path, symbols: int = 2, days: int = 400) -> None:
    """Write a synthetic daily trades CSV; 400 days leaves the 10% validation and test splits a full sequence."""
    rng = np.random.default_rng(0)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for symbol in range(symbols):
        prices = 100 + np.cumsum(rng.normal(0, 1, days))
        for day, price in enumerate(prices.tolist()):
            rows.append(
                {
                    "symbolId": f"symbol-{symbol}",
                    "bucket": (start + timedelta(days=day)).isoformat(),
                    "averagePrice": price,
                    "minPrice": price - 1,
                    "maxPrice": price + 1,
                    "volume": int(rng.integers(1, 1000)),
                }
            )
    pl.DataFrame(rows).write_csv(path)


@pytest.fixture
def clear_settings_cache() -> Iterator[None]:
    """Rebuild Settings from the patched environment, and drop them again afterwards."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("clear_settings_cache")
def test_plutus_forecasting_when_run_on_synthetic_data_should_write_a_loadable_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    data_path = tmp_path / "trades.csv"
    _write_trades_csv(data_path)
    output_dir = tmp_path / "models" / "smoke"
    output_dir.mkdir(parents=True)
    (output_dir / "tuned_params.json").write_text(json.dumps(_TUNED_PARAMS))
    monkeypatch.setattr("experiments.main.configure_logging", lambda settings: None)
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("PLUTUS_FORECAST_MODEL_NAME", "smoke")

    # Act
    main(
        [
            "plutus_forecasting",
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--no-retune",
            "--epochs",
            "2",
        ]
    )
    generator = ForecastGenerator()
    point = PlutusForecastPoint(average_price=100, min_price=99, max_price=101, volume=500)
    predictions = generator.predict_next([[point] * generator.sequence_length])

    # Assert
    assert {"model.pth", "params.json", "predictions.png"} <= {p.name for p in output_dir.iterdir()}
    assert len(predictions) == 1
