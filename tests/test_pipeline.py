import numpy as np
import pandas as pd

from market_regime.backtest import backtest_regime_strategy
from market_regime.data import synthetic_market_data
from market_regime.features import compute_regime_features, expanding_standardize
from market_regime.metrics import performance_metrics
from market_regime.models import GaussianHMMRegimeModel, KMeansRegimeModel, XGBoostRegimeClassifier
from market_regime.validation import walk_forward_predict


def test_features_are_finite_and_aligned():
    _, returns, _ = synthetic_market_data(300)
    features = compute_regime_features(returns)
    assert list(features) == ["downside_dev_10", "sortino_20", "sortino_60"]
    assert np.isfinite(features.to_numpy()).all()
    assert features.index.is_monotonic_increasing


def test_scaler_uses_train_only():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    test = pd.DataFrame({"x": [100.0]})
    scaled_train, scaled_test = expanding_standardize(train, test)
    assert np.isclose(scaled_train.mean().iloc[0], 0)
    assert scaled_test.iloc[0, 0] > 100


def test_backtest_enforces_one_day_delay_and_costs():
    index = pd.date_range("2020-01-01", periods=4)
    returns = pd.Series([0.1, 0.2, -0.1, 0.1], index=index)
    signal = pd.Series([1, 0, 1, 1], index=index)
    result = backtest_regime_strategy(returns, signal, transaction_cost_bps=10)
    assert result["position"].tolist() == [0, 1, 0, 1]
    assert result["cost"].sum() == 0.003


def test_walk_forward_kmeans_and_supervised_fallback():
    _, returns, _ = synthetic_market_data(450)
    features = compute_regime_features(returns)
    kmeans = walk_forward_predict(features, returns, KMeansRegimeModel, 200, 50)
    boosted = walk_forward_predict(
        features,
        returns,
        lambda: XGBoostRegimeClassifier(use_xgboost=False),
        200,
        50,
        supervised=True,
    )
    assert set(kmeans.unique()) <= {0, 1}
    assert kmeans.index.equals(boosted.index)


def test_metrics_have_expected_fields():
    returns = pd.Series([0.01, -0.02, 0.03] * 100)
    values = performance_metrics(returns, pd.Series(1.0, index=returns.index))
    assert {"cagr", "sharpe", "max_drawdown", "annual_turnover"} <= values.keys()


def test_hmm_causal_filter_runs_with_diagonal_covariance():
    rng = np.random.default_rng(11)
    X = np.r_[rng.normal(-1, 0.4, (80, 3)), rng.normal(1, 0.4, (80, 3))]
    model = GaussianHMMRegimeModel(n_iter=50).fit(X)
    states = model.predict_online(X[:10], initial_state=int(model.labels_[-1]))
    assert states.shape == (10,)
    assert set(states) <= {0, 1}
