# Market Regime Detection ML

Market regime detection using K-Means, Gaussian HMM, Statistical Jump Models (Temporally Regularized K-Means), and XGBoost with walk-forward backtesting.

## Model comparison

- **K-Means** — clustering with no temporal structure
- **Statistical Jump Model (Temporally Regularized K-Means / TR-KMeans)** — temporal persistence through an explicit regime-switching penalty
- **Gaussian HMM** — temporal persistence through transition probabilities
- **XGBoost** — supervised regime classification

“Temporally Regularized K-Means” is used here as a descriptive name for the discrete-state Statistical Jump Model, not as an established name from the literature.
