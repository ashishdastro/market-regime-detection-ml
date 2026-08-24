"""Annualized strategy and regime diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(returns: pd.Series, positions: pd.Series | None = None) -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        raise ValueError("returns must contain observations")
    annual = 252.0
    years = len(clean) / annual
    wealth = (1 + clean).cumprod()
    cagr = wealth.iloc[-1] ** (1 / years) - 1
    volatility = clean.std(ddof=1) * np.sqrt(annual)
    sharpe = clean.mean() / clean.std(ddof=1) * np.sqrt(annual) if clean.std(ddof=1) else np.nan
    downside = clean.clip(upper=0).pow(2).mean() ** 0.5 * np.sqrt(annual)
    sortino = clean.mean() * annual / downside if downside else np.nan
    drawdown = wealth / wealth.cummax() - 1
    maximum = float(drawdown.min())
    result = {
        "cagr": float(cagr),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": maximum,
        "calmar": float(cagr / abs(maximum)) if maximum else np.nan,
    }
    if positions is not None:
        aligned = positions.reindex(clean.index).ffill().fillna(0.0)
        result["annual_turnover"] = float(aligned.diff().abs().fillna(aligned.abs()).sum() / years)
        result["switches_per_year"] = float(aligned.diff().abs().fillna(0).gt(0).sum() / years)
    return result
