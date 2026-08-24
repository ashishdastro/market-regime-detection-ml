"""Publication-ready research figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_equity_curves(returns: pd.DataFrame, output: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wealth = (1 + returns.fillna(0)).cumprod()
    axis = wealth.plot(figsize=(11, 6), linewidth=1.5)
    axis.set(title="Out-of-sample equity curves", ylabel="Growth of $1", xlabel="")
    axis.grid(alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(output, dpi=180)
    plt.close(axis.figure)


def plot_lambda_sensitivity(results: pd.DataFrame, output: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, left = plt.subplots(figsize=(9, 5))
    ordered = results.sort_values("jump_penalty")
    left.plot(ordered["jump_penalty"], ordered["sharpe"], marker="o", label="Sharpe")
    left.set(xlabel="Jump penalty (lambda)", ylabel="Validation Sharpe")
    right = left.twinx()
    right.plot(
        ordered["jump_penalty"],
        ordered["switches_per_year"],
        color="tab:orange",
        marker="s",
        label="Switches/year",
    )
    right.set_ylabel("Switches per year")
    left.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
