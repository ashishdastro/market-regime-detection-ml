"""Paper-inspired market regime features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_regime_features(
    returns: pd.Series,
    risk_free: pd.Series | float = 0.0,
    downside_halflife: int = 10,
    sortino_halflives: tuple[int, ...] = (20, 60),
) -> pd.DataFrame:
    """Compute downside deviation and EWM Sortino features from excess returns."""
    if isinstance(risk_free, pd.Series):
        excess = returns.subtract(risk_free.reindex(returns.index).ffill().fillna(0.0))
    else:
        excess = returns - float(risk_free)
    downside_sq = excess.clip(upper=0).pow(2)

    def deviation(half_life: int) -> pd.Series:
        return downside_sq.ewm(halflife=half_life, adjust=False, min_periods=half_life).mean().pow(0.5)

    features = {"downside_dev_10": deviation(downside_halflife)}
    for half_life in sortino_halflives:
        mean = excess.ewm(halflife=half_life, adjust=False, min_periods=half_life).mean()
        features[f"sortino_{half_life}"] = mean.divide(deviation(half_life).replace(0, np.nan))
    return pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).dropna()


def expanding_standardize(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize a fold using training statistics only."""
    mean = train.mean()
    scale = train.std(ddof=0).replace(0, 1.0)
    return (train - mean) / scale, (test - mean) / scale

