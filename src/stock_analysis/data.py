from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from stock_analysis.config import ForecastConfig, StockDataConfig


DEFAULT_FEATURE_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class PreparedData:
    """Train/test arrays and their corresponding dates."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_dates: pd.Index
    gap_dates: pd.Index
    test_dates: pd.Index


class StockDataLoader:
    """Load historical market data from vnstock_data."""

    def __init__(self, config: StockDataConfig):
        self.config = config

    def load(self) -> pd.DataFrame:
        try:
            from vnstock_data import Quote
        except ImportError as exc:
            raise ImportError(
                "vnstock-data is required to download market data. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        quote = Quote(
            source=self.config.source,
            symbol=self.config.symbol,
            random_agent=False,
            show_log=False,
        )
        history = quote.history(
            start=self.config.start,
            end=self.config.end,
            interval=self.config.interval,
        )
        return normalize_market_frame(history)


class TimeSeriesPreprocessor:
    """Prepare OHLCV data for supervised time-series forecasting."""

    def __init__(
        self,
        config: ForecastConfig | None = None,
        feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    ):
        self.config = config or ForecastConfig()
        self.feature_columns = tuple(feature_columns)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_column_index = self.feature_columns.index(self.config.target_column)

    def prepare(self, data: str | Path | pd.DataFrame) -> PreparedData:
        frame = self._load_frame(data)
        train_raw, gap_raw, test_raw, dates = self._split(frame)
        train_scaled, _, test_scaled = self._scale(train_raw, gap_raw, test_raw)

        x_train, y_train = self._create_sequences(train_scaled)
        x_test, y_test = self._create_sequences(test_scaled)

        return PreparedData(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            train_dates=dates["train"],
            gap_dates=dates["gap"],
            test_dates=dates["test"],
        )

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values).reshape(-1)
        dummy = np.zeros((len(values), len(self.feature_columns)))
        dummy[:, self.target_column_index] = values
        return self.scaler.inverse_transform(dummy)[:, self.target_column_index]

    def _load_frame(self, data: str | Path | pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, (str, Path)):
            frame = pd.read_csv(data, index_col=0, parse_dates=True)
        else:
            frame = data.copy()

        frame = normalize_market_frame(frame)
        missing = sorted(set(self.feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        return frame.loc[:, self.feature_columns].dropna()

    def _split(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, pd.Index]]:
        values = frame.to_numpy(dtype=float)
        total_size = len(values)
        minimum_size = self.config.lookback + 10

        if total_size < minimum_size:
            raise ValueError(f"Not enough data: {total_size} rows. Need at least {minimum_size}.")

        train_size = int(total_size * self.config.train_ratio)
        gap_size = int(total_size * self.config.gap_ratio)
        test_size = total_size - train_size - gap_size

        if test_size <= self.config.lookback:
            raise ValueError("Test set is too small. Increase the date range or reduce lookback.")

        dates = {
            "train": frame.index[:train_size],
            "gap": frame.index[train_size : train_size + gap_size],
            "test": frame.index[train_size + gap_size :],
        }

        return (
            values[:train_size],
            values[train_size : train_size + gap_size],
            values[train_size + gap_size :],
            dates,
        )

    def _scale(
        self,
        train_data: np.ndarray,
        gap_data: np.ndarray,
        test_data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.scaler.fit(train_data)
        return (
            self.scaler.transform(train_data),
            self.scaler.transform(gap_data),
            self.scaler.transform(test_data),
        )

    def _create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_values, y_values = [], []

        for index in range(len(data) - self.config.lookback):
            x_values.append(data[index : index + self.config.lookback])
            y_values.append(data[index + self.config.lookback, self.target_column_index])

        return np.asarray(x_values), np.asarray(y_values)


def normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common vnstock column variants into a clean OHLCV frame."""

    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    if "time" in normalized.columns:
        normalized["time"] = pd.to_datetime(normalized["time"])
        normalized = normalized.set_index("time")

    normalized.index = pd.to_datetime(normalized.index)
    return normalized.sort_index()

