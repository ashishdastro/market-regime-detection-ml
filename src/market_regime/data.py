"""Public market-data ingestion and deterministic synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def download_prices(symbol: str, start: str, end: str | None = None) -> pd.Series:
    """Download adjusted daily prices from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the 'market-data' extra to download Yahoo data") from exc
    frame = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if frame.empty:
        raise ValueError(f"No data returned for {symbol}")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename(symbol).dropna()


def download_risk_free(start: str, end: str | None = None) -> pd.Series:
    """Download the 3-month Treasury yield and convert it to a daily simple rate."""
    try:
        from pandas_datareader import data as web
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the 'market-data' extra to download FRED data") from exc
    annual_percent = web.DataReader("DGS3MO", "fred", start, end)["DGS3MO"].ffill()
    return ((1 + annual_percent / 100) ** (1 / 252) - 1).rename("risk_free")


def synthetic_market_data(
    n_days: int = 2500, random_state: int = 42
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Generate a reproducible two-regime series for demos and CI."""
    rng = np.random.default_rng(random_state)
    transition = np.array([[0.985, 0.015], [0.04, 0.96]])
    states = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        states[t] = rng.choice(2, p=transition[states[t - 1]])
    means = np.array([0.0006, -0.0010])
    vols = np.array([0.009, 0.022])
    returns = rng.normal(means[states], vols[states])
    index = pd.bdate_range("2010-01-04", periods=n_days)
    series = pd.Series(returns, index=index, name="return")
    prices = (100 * (1 + series).cumprod()).rename("SYNTH")
    return prices, series, pd.Series(states, index=index, name="true_state")

