# Market Regime Detection Using Machine Learning

Market regime detection using K-Means, Gaussian HMM, Statistical Jump Models
(Temporally Regularized K-Means), and XGBoost with leakage-safe walk-forward backtesting.

This project asks: **does explicit temporal persistence improve financial regime identification?**

| Model | Learning setup | Temporal structure |
|---|---|---|
| K-Means | Unsupervised clustering | None |
| Statistical Jump Model / TR-KMeans | Unsupervised clustering | Explicit switching penalty |
| Gaussian HMM | Latent-state probabilistic model | Learned transition probabilities |
| XGBoost | Supervised regime classification | Nonlinear feature interactions |

> **Naming note:** “Temporally Regularized K-Means” (TR-KMeans) is this project's descriptive
> name for the discrete-state **Statistical Jump Model**. It is not presented as an established
> name in the literature.

## Statistical Jump Model from scratch

The centerpiece is a from-scratch implementation of

\[
\min_{\theta,s}\sum_{t=1}^{T}\frac{1}{2}\|x_t-\theta_{s_t}\|^2
+\lambda\sum_{t=2}^{T}\mathbf{1}(s_t\ne s_{t-1}).
\]

For fixed regimes, centroids are sample means. For fixed centroids, the globally optimal state path
is recovered with a Viterbi-like dynamic program. The implementation alternates these updates
across K-Means++ initializations and exposes both offline decoding and strictly causal online
filtering. Setting `jump_penalty=0` recovers the K-Means-like case; increasing it makes regime
changes more expensive.

## Research design

The default features reproduce the paper's compact excess-return feature set:

- exponentially weighted downside deviation, 10-day half-life;
- exponentially weighted Sortino ratio, 20-day half-life;
- exponentially weighted Sortino ratio, 60-day half-life.

Evaluation uses expanding walk-forward folds. Scaling, model fitting, cluster naming, and supervised
targets use the training window only. HMM states use a causal forward filter—not a hindsight Viterbi
path. XGBoost predicts whether the next 21-session return is positive; labels whose outcome would
cross a fold boundary are excluded. Most importantly, the signal at time \(t\) controls the position
at \(t+1\). Transaction costs are charged on position changes.

Diagnostics include CAGR, volatility, Sharpe, Sortino, maximum drawdown, Calmar ratio, annual
turnover, and regime switches per year.

For long monthly studies, `--max-train-size 2016` uses a rolling eight-year training window. This
keeps the model adaptive and the from-scratch dynamic program computationally reproducible while
retaining strict train-before-test ordering.

## Repository structure

```text
src/market_regime/
  data.py          Yahoo/FRED ingestion and deterministic synthetic data
  features.py      downside-deviation and Sortino features
  jump_model.py    from-scratch dynamic-programming Statistical Jump Model
  models.py        K-Means, causal Gaussian HMM, and XGBoost adapters
  validation.py    fold-local scaling, walk-forward prediction, lambda tuning
  backtest.py      one-day-delayed allocation and transaction costs
  metrics.py       return, risk, drawdown, and turnover diagnostics
  plotting.py      equity-curve and lambda-sensitivity figures
  experiment.py    reproducible end-to-end comparison
tests/              algorithmic, leakage, HMM, and pipeline tests
scripts/            command-line research entry point
```

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -e ".[all]"
python -m pytest
```

Run the complete pipeline offline on deterministic synthetic data:

```bash
python -m market_regime.cli --synthetic --symbol SYNTH
```

Run it on public Yahoo Finance data:

```bash
python -m market_regime.cli --symbol SPY --start 1993-01-29
python -m market_regime.cli --symbol QQQ --start 1999-03-10
python -m market_regime.cli --symbol BTC-USD --start 2014-09-17
```

Results are written to `reports/`: daily backtests, a metrics table, metadata, and an out-of-sample
equity-curve figure. Generated artifacts and raw data are ignored by Git.

## Lambda sensitivity

`tune_jump_penalty` selects the switching penalty on a trailing validation window using delayed,
cost-adjusted strategy Sharpe. `plot_lambda_sensitivity` shows both validation Sharpe and switches
per year, making the persistence/performance trade-off explicit.

```python
from market_regime.plotting import plot_lambda_sensitivity
from market_regime.validation import tune_jump_penalty

study = tune_jump_penalty(features, returns, penalties=[0, 1, 5, 10, 25, 50, 100])
plot_lambda_sensitivity(study, "figures/lambda_sensitivity.png")
```

## Reproducibility and limitations

- Random seeds are fixed by default.
- Synthetic data keeps CI and local verification independent of network availability.
- Yahoo Finance and FRED are public substitutes for the paper's proprietary data.
- Results will differ with indices, rates, costs, dates, or data vendors.
- This is research software, not investment advice or a production trading system.

## Reference

Inspired by Shu, Yu, and Mulvey, *Downside Risk Reduction Using Regime-Switching Signals: A
Statistical Jump Model Approach* ([arXiv:2402.05272](https://arxiv.org/abs/2402.05272)).

See [paper/results.md](paper/results.md) for the SPY, QQQ, and BTC-USD walk-forward findings.
