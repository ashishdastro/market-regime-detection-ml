"""End-to-end reproducible research experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backtest import backtest_regime_strategy
from .data import download_prices, synthetic_market_data
from .features import compute_regime_features
from .jump_model import JumpModel
from .metrics import performance_metrics
from .models import GaussianHMMRegimeModel, KMeansRegimeModel, XGBoostRegimeClassifier
from .plotting import plot_equity_curves
from .validation import walk_forward_predict


def run_experiment(
    symbol: str = "SPY",
    start: str = "1990-01-01",
    synthetic: bool = False,
    output_dir: str | Path = "reports",
    min_train_size: int = 756,
    test_size: int = 21,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    if synthetic:
        prices, returns, _ = synthetic_market_data()
    else:
        prices = download_prices(symbol, start)
        returns = prices.pct_change().dropna()
    features = compute_regime_features(returns)
    aligned_returns = returns.reindex(features.index)
    factories = {
        "kmeans": lambda: KMeansRegimeModel(),
        "tr_kmeans": lambda: JumpModel(jump_penalty=1.0, n_init=5),
        "gaussian_hmm": lambda: GaussianHMMRegimeModel(),
        "xgboost": lambda: XGBoostRegimeClassifier(),
    }
    rows = []
    strategy_returns = {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, factory in factories.items():
        signal = walk_forward_predict(
            features,
            aligned_returns,
            factory,
            min_train_size=min_train_size,
            test_size=test_size,
            supervised=name == "xgboost",
        )
        result = backtest_regime_strategy(
            aligned_returns.reindex(signal.index), signal, transaction_cost_bps=transaction_cost_bps
        )
        result.to_csv(output / f"{symbol.lower()}_{name}_backtest.csv")
        strategy_returns[name] = result["strategy_return"]
        rows.append({"model": name, **performance_metrics(result["strategy_return"], result["position"])})
    benchmark = aligned_returns.iloc[min_train_size:]
    strategy_returns["buy_and_hold"] = benchmark
    rows.append({"model": "buy_and_hold", **performance_metrics(benchmark, pd.Series(1.0, index=benchmark.index))})
    metrics = pd.DataFrame(rows).set_index("model").sort_values("sharpe", ascending=False)
    metrics.to_csv(output / f"{symbol.lower()}_metrics.csv")
    plot_equity_curves(pd.DataFrame(strategy_returns), output / f"{symbol.lower()}_equity_curves.png")
    (output / f"{symbol.lower()}_metadata.json").write_text(
        json.dumps({"symbol": symbol, "start": start, "synthetic": synthetic, "observations": len(features)}, indent=2),
        encoding="utf-8",
    )
    return metrics
