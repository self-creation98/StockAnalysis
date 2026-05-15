import numpy as np
import pandas as pd

from stock_analysis.config import ForecastConfig
from stock_analysis.data import TimeSeriesPreprocessor, normalize_market_frame


def test_normalize_market_frame_uses_time_index():
    frame = pd.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-01"],
            "Open": [11, 10],
            "High": [12, 11],
            "Low": [10, 9],
            "Close": [11.5, 10.5],
            "Volume": [2000, 1000],
        }
    )

    normalized = normalize_market_frame(frame)

    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert normalized.index.is_monotonic_increasing


def test_preprocessor_creates_expected_sequence_shapes():
    rows = 80
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "open": np.linspace(10, 20, rows),
            "high": np.linspace(11, 21, rows),
            "low": np.linspace(9, 19, rows),
            "close": np.linspace(10.5, 20.5, rows),
            "volume": np.arange(rows) + 1000,
        }
    )

    prepared = TimeSeriesPreprocessor(ForecastConfig(lookback=5)).prepare(frame)

    assert prepared.x_train.shape == (59, 5, 5)
    assert prepared.y_train.shape == (59,)
    assert prepared.x_test.shape == (3, 5, 5)
    assert prepared.y_test.shape == (3,)

