# StockAnalysis - Vietnam Stock Market

StockAnalysis is a Python project for Vietnam stock market analysis and forecasting. The core logic has been moved out of the exploratory notebook into a reusable package so the project is easier to run, test, and extend.

## Current Features

- Download historical prices with `vnstock-data`.
- Normalize OHLCV data and create supervised time-series windows.
- Split data into 80% train, 10% gap, and 10% test to reduce data leakage.
- Train a stacking ensemble with ARIMA, Random Forest, and an XGBoost meta-learner.
- Optionally add LSTM, GRU, and Transformer base learners when TensorFlow is installed.

## Project Structure

```text
StockAnalysis/
|-- main.ipynb                 # Original exploratory notebook
|-- pyproject.toml             # Package configuration
|-- requirements.txt           # Core dependencies
`-- src/
    `-- stock_analysis/
        |-- cli.py             # Pipeline entry point
        |-- config.py          # Data and forecast configuration
        |-- data.py            # Load, clean, split, and scale data
        |-- ensemble.py        # Stacking ensemble
        |-- metrics.py         # Regression metrics
        `-- models.py          # Base learners
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

To enable LSTM, GRU, and Transformer models:

```bash
pip install -e ".[deep-learning]"
```

## Usage

```bash
stock-analysis --symbol HPG --start 2017-01-01 --end 2022-12-31
```

You can also run the package as a module:

```bash
python -m stock_analysis --symbol HPG --start 2017-01-01 --end 2022-12-31
```

Run with deep learning base learners:

```bash
stock-analysis --symbol HPG --include-deep-learning
```

## Roadmap

- Add technical indicators such as RSI, MACD, and Bollinger Bands.
- Add forecast evaluation charts.
- Save trained models and run outputs under `artifacts/`.
- Build an interactive Streamlit dashboard.
