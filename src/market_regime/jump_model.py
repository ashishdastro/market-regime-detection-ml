"""Statistical Jump Model (Temporally Regularized K-Means).

The discrete-state objective is

    sum_t 0.5 * ||x_t - centroid[state_t]||^2
    + jump_penalty * sum_t I(state_t != state_{t-1}).

Centroids and the globally optimal state path are alternately updated. The
path update is solved exactly with dynamic programming.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class FitDiagnostics:
    objective: float
    n_iter: int
    converged: bool


class JumpModel:
    """Discrete Statistical Jump Model with a scikit-learn-style API."""

    def __init__(
        self,
        n_components: int = 2,
        jump_penalty: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        n_init: int = 10,
        random_state: int | None = 42,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be positive")
        if jump_penalty < 0:
            raise ValueError("jump_penalty must be non-negative")
        self.n_components = n_components
        self.jump_penalty = float(jump_penalty)
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    @staticmethod
    def _validate_x(X: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite values")
        return values

    def _initialize_centroids(
        self, X: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """K-Means++ initialization without a scikit-learn dependency."""
        centers = [X[rng.integers(len(X))]]
        while len(centers) < self.n_components:
            distances = np.min(
                np.stack([np.sum((X - center) ** 2, axis=1) for center in centers]), axis=0
            )
            total = distances.sum()
            index = rng.integers(len(X)) if total <= 0 else rng.choice(len(X), p=distances / total)
            centers.append(X[index])
        return np.asarray(centers, dtype=float)

    def _emission_costs(self, X: NDArray[np.float64], centers: NDArray[np.float64]):
        return 0.5 * np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    def _optimal_path(
        self,
        costs: NDArray[np.float64],
        initial_state: int | None = None,
    ) -> tuple[NDArray[np.int64], float]:
        """Return the globally optimal state sequence for fixed centroids."""
        n_samples, n_states = costs.shape
        dp = np.empty_like(costs)
        back = np.zeros((n_samples, n_states), dtype=np.int64)
        dp[0] = costs[0]
        if initial_state is not None:
            dp[0] += self.jump_penalty * (np.arange(n_states) != initial_state)
        transition = self.jump_penalty * (1 - np.eye(n_states))
        for t in range(1, n_samples):
            candidates = dp[t - 1][:, None] + transition
            back[t] = np.argmin(candidates, axis=0)
            dp[t] = costs[t] + candidates[back[t], np.arange(n_states)]
        states = np.empty(n_samples, dtype=np.int64)
        states[-1] = np.argmin(dp[-1])
        for t in range(n_samples - 1, 0, -1):
            states[t - 1] = back[t, states[t]]
        return states, float(dp[-1, states[-1]])

    def _updated_centroids(
        self,
        X: NDArray[np.float64],
        states: NDArray[np.int64],
        old: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        centers = old.copy()
        for state in range(self.n_components):
            members = X[states == state]
            centers[state] = members.mean(axis=0) if len(members) else X[rng.integers(len(X))]
        return centers

    def fit(self, X: ArrayLike) -> JumpModel:
        values = self._validate_x(X)
        if len(values) < self.n_components:
            raise ValueError("n_samples must be at least n_components")
        master_rng = np.random.default_rng(self.random_state)
        best: tuple[float, NDArray[np.float64], NDArray[np.int64], int, bool] | None = None
        for _ in range(self.n_init):
            rng = np.random.default_rng(master_rng.integers(2**32))
            centers = self._initialize_centroids(values, rng)
            previous = np.inf
            converged = False
            for iteration in range(1, self.max_iter + 1):
                states, _ = self._optimal_path(self._emission_costs(values, centers))
                centers = self._updated_centroids(values, states, centers, rng)
                states, objective = self._optimal_path(self._emission_costs(values, centers))
                if np.isfinite(previous) and abs(previous - objective) <= self.tol * max(
                    1.0, abs(previous)
                ):
                    converged = True
                    break
                previous = objective
            candidate = (objective, centers.copy(), states.copy(), iteration, converged)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        objective, self.cluster_centers_, self.labels_, n_iter, converged = best
        self.diagnostics_ = FitDiagnostics(objective, n_iter, converged)
        self.n_features_in_ = values.shape[1]
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("fit must be called before prediction")

    def predict(self, X: ArrayLike, initial_state: int | None = None) -> NDArray[np.int64]:
        """Offline decoding: optimize the entire supplied sequence jointly."""
        self._check_fitted()
        values = self._validate_x(X)
        return self._optimal_path(self._emission_costs(values, self.cluster_centers_), initial_state)[0]

    def predict_online(
        self, X: ArrayLike, initial_state: int | None = None
    ) -> NDArray[np.int64]:
        """Causal filtering that never revises a state after observing future rows."""
        self._check_fitted()
        costs = self._emission_costs(self._validate_x(X), self.cluster_centers_)
        filtered = np.empty(len(costs), dtype=np.int64)
        cumulative = costs[0].copy()
        if initial_state is not None:
            cumulative += self.jump_penalty * (np.arange(self.n_components) != initial_state)
        filtered[0] = np.argmin(cumulative)
        transition = self.jump_penalty * (1 - np.eye(self.n_components))
        for t in range(1, len(costs)):
            cumulative = costs[t] + np.min(cumulative[:, None] + transition, axis=0)
            filtered[t] = np.argmin(cumulative)
        return filtered

    def score(self, X: ArrayLike) -> float:
        self._check_fitted()
        values = self._validate_x(X)
        _, objective = self._optimal_path(self._emission_costs(values, self.cluster_centers_))
        return -objective
