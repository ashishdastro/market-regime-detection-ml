# Out-of-sample results

Run date: 2026-08-24. Prices were downloaded from Yahoo Finance. Every model used a 756-observation
minimum training period, an eight-year/2,016-observation rolling training window, 21-observation
test blocks, a one-observation execution delay, and 5 bps transaction costs per unit of turnover.
Equities use 252-session annualization and BTC-USD uses 365-day annualization.

## SPY

8,388 feature observations from 1993-01-29 onward.

| Model | CAGR | Sharpe | Max drawdown | Switches/year |
|---|---:|---:|---:|---:|
| Buy and hold | **10.37%** | **0.609** | -55.19% | 0.00 |
| XGBoost | 8.51% | 0.558 | -40.48% | 23.67 |
| TR-KMeans | 3.32% | 0.432 | -20.03% | 6.17 |
| Gaussian HMM | 2.84% | 0.388 | **-18.46%** | 5.45 |
| K-Means | 2.92% | 0.385 | -22.39% | 12.78 |

Buy and hold remained strongest on return and Sharpe. Temporal models substantially reduced
drawdown, and both TR-KMeans and the HMM switched roughly half as often as K-Means, but their low
risky-asset exposure sacrificed too much upside.

## QQQ

6,846 feature observations from 1999-03-10 onward.

| Model | CAGR | Sharpe | Max drawdown | Switches/year |
|---|---:|---:|---:|---:|
| XGBoost | 13.64% | **0.774** | -36.23% | 26.77 |
| Buy and hold | **15.10%** | 0.741 | -53.40% | 0.00 |
| K-Means | 7.06% | 0.583 | **-26.63%** | 13.37 |
| Gaussian HMM | 5.53% | 0.495 | -27.03% | 6.99 |
| TR-KMeans | 5.49% | 0.472 | -29.93% | 7.74 |

XGBoost delivered the best Sharpe and improved drawdown versus buy and hold, but at high turnover.
K-Means produced the smallest drawdown. Temporal persistence again roughly halved the switching
rate relative to K-Means, without improving risk-adjusted returns in this asset.

## BTC-USD

4,300 feature observations from 2014-09-17 onward.

| Model | CAGR | Sharpe | Max drawdown | Switches/year |
|---|---:|---:|---:|---:|
| Gaussian HMM | 57.90% | **1.224** | **-51.78%** | 7.31 |
| TR-KMeans | 53.94% | 1.203 | -62.10% | 7.93 |
| K-Means | 51.77% | 1.169 | -52.53% | 19.47 |
| Buy and hold | **61.01%** | 1.045 | -83.40% | 0.00 |
| XGBoost | 32.08% | 0.785 | -82.06% | 51.19 |

BTC supplies the strongest evidence for temporal regime modeling. The HMM and TR-KMeans both
improved Sharpe over buy and hold and K-Means while switching far less often than K-Means. The HMM
also reduced maximum drawdown by more than 31 percentage points versus buy and hold.

## Cross-asset conclusion

The results support a nuanced answer to the research question. Explicit persistence consistently
reduces regime churn: TR-KMeans switched 52% less often than K-Means in SPY, 42% less in QQQ, and
59% less in BTC-USD. That persistence translated into superior risk-adjusted performance in BTC,
but not in the two equity ETFs. The method is therefore useful as a structural regularizer, not a
guaranteed alpha source. XGBoost helped on QQQ but overtraded on SPY and BTC, illustrating why
turnover and cost-aware evaluation belong in the model comparison.

