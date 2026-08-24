"""Causal adapters for comparison models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier


class KMeansRegimeModel:
    def __init__(self, n_components: int = 2, random_state: int = 42) -> None:
        self.model = KMeans(n_clusters=n_components, n_init=20, random_state=random_state)

    def fit(self, X: NDArray[np.float64]) -> KMeansRegimeModel:
        self.model.fit(X)
        self.labels_ = self.model.labels_
        return self

    def predict_online(self, X: NDArray[np.float64], initial_state=None) -> NDArray[np.int64]:
        return self.model.predict(X)


class GaussianHMMRegimeModel:
    """Gaussian HMM fitted by hmmlearn and inferred with a causal forward filter."""

    def __init__(self, n_components: int = 2, random_state: int = 42, n_iter: int = 200):
        self.n_components = n_components
        self.random_state = random_state
        self.n_iter = n_iter

    def fit(self, X: NDArray[np.float64]) -> GaussianHMMRegimeModel:
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install the 'models' extra to run the Gaussian HMM") from exc
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=self.random_state,
        ).fit(X)
        self.labels_ = self._filter(X)
        return self

    def _log_emissions(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        means = self.model.means_
        # The public property expands diagonal covariances into full matrices;
        # the fitted private array preserves the (state, feature) representation.
        variances = np.maximum(np.asarray(self.model._covars_), 1e-9)
        return -0.5 * np.sum(
            np.log(2 * np.pi * variances)[None, :, :] + (X[:, None, :] - means) ** 2 / variances,
            axis=2,
        )

    def _filter(self, X: NDArray[np.float64], initial_state: int | None = None):
        emissions = self._log_emissions(X)
        probabilities = np.zeros((len(X), self.n_components))
        prior = self.model.startprob_.copy()
        if initial_state is not None:
            prior = np.eye(self.n_components)[initial_state]
        for t, log_likelihood in enumerate(emissions):
            if t:
                prior = probabilities[t - 1] @ self.model.transmat_
            posterior = prior * np.exp(log_likelihood - log_likelihood.max())
            probabilities[t] = posterior / posterior.sum()
        return probabilities.argmax(axis=1)

    def predict_online(self, X: NDArray[np.float64], initial_state=None):
        return self._filter(X, initial_state)


class XGBoostRegimeClassifier:
    """Supervised classifier with a deterministic sklearn fallback."""

    def __init__(self, random_state: int = 42, use_xgboost: bool = True) -> None:
        self.random_state = random_state
        self.use_xgboost = use_xgboost

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]):
        if self.use_xgboost:
            try:
                from xgboost import XGBClassifier

                self.model = XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=self.random_state,
                    n_jobs=1,
                )
            except ImportError:  # pragma: no cover
                self.model = HistGradientBoostingClassifier(max_iter=200, random_state=self.random_state)
        else:
            self.model = HistGradientBoostingClassifier(max_iter=200, random_state=self.random_state)
        self.model.fit(X, y)
        return self

    def predict_online(self, X: NDArray[np.float64], initial_state=None):
        return self.model.predict(X).astype(np.int64)
