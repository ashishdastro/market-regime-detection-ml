"""Expanding-window, leakage-safe regime evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from .features import expanding_standardize
from .jump_model import JumpModel
from .models import XGBoostRegimeClassifier


def bull_state(labels: np.ndarray, returns: pd.Series) -> int:
    """Name states using only the in-sample mean return."""
    values = returns.to_numpy()
    means = [values[labels == state].mean() if np.any(labels == state) else -np.inf for state in np.unique(labels)]
    return int(np.unique(labels)[int(np.argmax(means))])


def walk_forward_predict(
    features: pd.DataFrame,
    returns: pd.Series,
    model_factory: Callable[[], object],
    min_train_size: int = 756,
    test_size: int = 21,
    supervised: bool = False,
    label_horizon: int = 21,
) -> pd.Series:
    """Retrain on an expanding window and causally infer each test block."""
    features = features.sort_index()
    returns = returns.reindex(features.index)
    predictions = pd.Series(index=features.index, dtype=float, name="bull_signal")
    for start in range(min_train_size, len(features), test_size):
        stop = min(start + test_size, len(features))
        train_x, test_x = expanding_standardize(features.iloc[:start], features.iloc[start:stop])
        train_returns = returns.iloc[:start]
        model = model_factory()
        if supervised:
            if not isinstance(model, XGBoostRegimeClassifier):
                raise TypeError("supervised=True requires XGBoostRegimeClassifier")
            # Every target is fully realized before the fold boundary. This
            # turns XGBoost into a genuine regime forecaster without allowing
            # test-period returns to leak into its training labels.
            forward_return = (
                (1 + train_returns)
                .rolling(label_horizon)
                .apply(np.prod, raw=True)
                .shift(-label_horizon)
                .sub(1)
                .dropna()
            )
            supervised_x = train_x.reindex(forward_return.index)
            y = (forward_return > 0).astype(np.int64)
            if y.nunique() < 2:
                y = (forward_return > forward_return.median()).astype(np.int64)
            model.fit(supervised_x.to_numpy(), y.to_numpy())
            test_states = model.predict_online(test_x.to_numpy())
            predictions.iloc[start:stop] = test_states
        else:
            model.fit(train_x.to_numpy())
            train_labels = np.asarray(model.labels_)
            in_sample_bull = bull_state(train_labels, train_returns)
            initial = int(train_labels[-1])
            test_states = model.predict_online(test_x.to_numpy(), initial_state=initial)
            predictions.iloc[start:stop] = (test_states == in_sample_bull).astype(float)
    return predictions.dropna().astype(int)


def tune_jump_penalty(
    features: pd.DataFrame,
    returns: pd.Series,
    penalties: Iterable[float],
    validation_size: int = 504,
) -> pd.DataFrame:
    """Tune lambda on a trailing validation set using delayed strategy Sharpe."""
    from .backtest import backtest_regime_strategy
    from .metrics import performance_metrics

    if len(features) <= validation_size:
        raise ValueError("features must be longer than validation_size")
    train_x = features.iloc[:-validation_size]
    validation_x = features.iloc[-validation_size:]
    train_scaled, validation_scaled = expanding_standardize(train_x, validation_x)
    rows = []
    for penalty in penalties:
        model = JumpModel(jump_penalty=penalty, n_init=5).fit(train_scaled.to_numpy())
        bull = bull_state(model.labels_, returns.reindex(train_x.index))
        states = model.predict_online(validation_scaled.to_numpy(), int(model.labels_[-1]))
        signal = pd.Series(states == bull, index=validation_x.index)
        result = backtest_regime_strategy(returns.reindex(validation_x.index), signal)
        metrics = performance_metrics(result["strategy_return"], result["position"])
        rows.append({"jump_penalty": penalty, **metrics})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
