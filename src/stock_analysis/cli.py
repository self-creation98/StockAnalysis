import argparse

from stock_analysis.config import ForecastConfig, StockDataConfig
from stock_analysis.data import StockDataLoader, TimeSeriesPreprocessor
from stock_analysis.ensemble import StackingEnsembleForecaster
from stock_analysis.models import (
    default_model_specs,
    deep_learning_model_specs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Vietnam stock forecasting pipeline.")
    parser.add_argument("--symbol", default="HPG", help="Ticker symbol, for example HPG.")
    parser.add_argument("--source", default="vci", help="vnstock data source.")
    parser.add_argument("--start", default="2017-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default="2022-12-31", help="End date in YYYY-MM-DD format.")
    parser.add_argument("--lookback", type=int, default=10, help="Number of historical rows per sample.")
    parser.add_argument("--splits", type=int, default=5, help="Number of time-series CV splits.")
    parser.add_argument(
        "--include-deep-learning",
        action="store_true",
        help="Add LSTM, GRU and Transformer base learners. Requires tensorflow-cpu.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_config = StockDataConfig(
        symbol=args.symbol,
        source=args.source,
        start=args.start,
        end=args.end,
    )
    forecast_config = ForecastConfig(lookback=args.lookback, n_splits=args.splits)

    print(f"Loading {data_config.symbol} data from {data_config.start} to {data_config.end}...")
    frame = StockDataLoader(data_config).load()

    preprocessor = TimeSeriesPreprocessor(forecast_config)
    prepared = preprocessor.prepare(frame)

    model_specs = default_model_specs(random_state=forecast_config.random_state)
    if args.include_deep_learning:
        model_specs += deep_learning_model_specs()

    print(f"Training ensemble with {', '.join(spec.name for spec in model_specs)}...")
    forecaster = StackingEnsembleForecaster(model_specs=model_specs, n_splits=forecast_config.n_splits)
    forecaster.fit(prepared.x_train, prepared.y_train)

    print("\nEvaluation")
    print("-" * 60)
    results = forecaster.evaluate(
        prepared.x_test,
        prepared.y_test,
        inverse_transform=preprocessor.inverse_transform_target,
    )
    for name, metrics in results.items():
        print(f"{name:18s} RMSE={metrics.rmse:,.2f} MAE={metrics.mae:,.2f} R2={metrics.r2:.4f}")

    importance = forecaster.feature_importance()
    if importance:
        print("\nMeta-learner feature importance")
        print("-" * 60)
        for model_name, score in importance.items():
            print(f"{model_name:18s} {score:.4f}")

