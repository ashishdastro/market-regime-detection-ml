# Reproducible experiments

The command-line runner is the canonical experiment interface so every analysis is reproducible in
CI and outside a notebook.

```bash
python -m market_regime.cli --synthetic --symbol SYNTH
python -m market_regime.cli --symbol SPY --start 1993-01-29
python -m market_regime.cli --symbol QQQ --start 1999-03-10
python -m market_regime.cli --symbol BTC-USD --start 2014-09-17
python -m market_regime.cli --symbol ETH-USD --start 2017-11-09
```

Use the synthetic run as the offline smoke test. Public-data experiments are intentionally not
committed because vendors can revise price history; metadata records the symbol, dates, and sample
size for each run.

