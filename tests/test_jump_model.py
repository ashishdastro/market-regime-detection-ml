import numpy as np

from market_regime.jump_model import JumpModel


def test_dynamic_programming_matches_brute_force():
    model = JumpModel(n_components=2, jump_penalty=1.3)
    costs = np.array([[0.1, 1.0], [0.4, 0.2], [0.3, 0.8], [1.2, 0.1]])
    states, objective = model._optimal_path(costs)
    candidates = []
    for encoded in range(2 ** len(costs)):
        path = np.array([(encoded >> i) & 1 for i in range(len(costs))])
        value = costs[np.arange(len(costs)), path].sum() + 1.3 * np.diff(path).astype(bool).sum()
        candidates.append((value, path))
    expected_value, expected_path = min(candidates, key=lambda item: item[0])
    assert np.isclose(objective, expected_value)
    assert np.array_equal(states, expected_path)


def test_jump_penalty_reduces_switching():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 1))
    low = JumpModel(jump_penalty=0, n_init=5).fit(X)
    high = JumpModel(jump_penalty=20, n_init=5).fit(X)
    assert np.diff(high.labels_).astype(bool).sum() <= np.diff(low.labels_).astype(bool).sum()


def test_fit_is_reproducible_and_predicts():
    X = np.r_[np.full((30, 2), -2.0), np.full((30, 2), 2.0)]
    first = JumpModel(jump_penalty=2, random_state=7).fit(X)
    second = JumpModel(jump_penalty=2, random_state=7).fit(X)
    assert np.allclose(first.cluster_centers_, second.cluster_centers_)
    assert len(first.predict_online(X)) == len(X)

