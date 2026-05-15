from dataclasses import dataclass


@dataclass(frozen=True)
class StockDataConfig:
    """Configuration for historical stock data retrieval."""

    symbol: str = "HPG"
    source: str = "vci"
    start: str = "2017-01-01"
    end: str = "2022-12-31"
    interval: str = "1D"


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for the forecasting pipeline."""

    lookback: int = 10
    train_ratio: float = 0.8
    gap_ratio: float = 0.1
    target_column: str = "close"
    random_state: int = 42
    n_splits: int = 5

