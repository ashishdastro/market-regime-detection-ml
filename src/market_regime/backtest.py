"""Leakage-safe delayed-signal backtesting."""

from __future__ import annotations

import pandas as pd


def backtest_regime_strategy(
    risky_returns: pd.Series,
    bull_signal: pd.Series,
    risk_free: pd.Series | float = 0.0,
    transaction_cost_bps: float = 5.0,
    trading_delay: int = 1,
) -> pd.DataFrame:
    """Apply today's signal only after ``trading_delay`` observations."""
    if trading_delay < 1:
        raise ValueError("trading_delay must be at least one to prevent same-day look-ahead")
    signal = bull_signal.reindex(risky_returns.index).ffill().fillna(0).astype(float).clip(0, 1)
    position = signal.shift(trading_delay).fillna(0.0)
    rf = risk_free.reindex(risky_returns.index).ffill().fillna(0.0) if isinstance(risk_free, pd.Series) else float(risk_free)
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * transaction_cost_bps / 10_000
    strategy = position * risky_returns + (1 - position) * rf - cost
    return pd.DataFrame(
        {"risky_return": risky_returns, "signal": signal, "position": position, "turnover": turnover, "cost": cost, "strategy_return": strategy}
    ).dropna(subset=["risky_return"])

