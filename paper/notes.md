# Research notes

Primary reference: Shu, Yu, and Mulvey, *Downside Risk Reduction Using Regime-Switching Signals:
A Statistical Jump Model Approach* ([arXiv:2402.05272](https://arxiv.org/abs/2402.05272)).

## Reproduction choices

- Public adjusted-close data replaces proprietary total-return indices.
- The three core features preserve the paper's 10/20/60-day exponential half-lives.
- Two regimes are named bull and bear using in-sample mean returns only.
- Online predictions and a one-observation trade delay prevent hindsight signal execution.
- The jump penalty is selected on validation strategy Sharpe when sensitivity tuning is used.
- Transaction costs apply to absolute changes in risky-asset weight.

## Extension

The same pipeline is intentionally reusable for SPY, QQQ, BTC-USD, and ETH-USD. This tests whether
explicit regime persistence generalizes from equities to cryptocurrency without changing the
algorithm or inventing asset-specific features.

## Interpretation

K-Means and the Statistical Jump Model form a controlled comparison: at zero penalty the temporal
regularizer disappears. The Gaussian HMM supplies a probabilistic form of persistence, while
XGBoost tests whether nonlinear supervised classification can forecast a positive 21-session
return. Training rows whose forward label would cross the fold boundary are removed.
