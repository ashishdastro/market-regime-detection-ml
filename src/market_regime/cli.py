"""Command-line entry point."""

from __future__ import annotations

import argparse

from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe market regime research")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--synthetic", action="store_true", help="Use deterministic offline data")
    parser.add_argument("--min-train-size", type=int, default=756)
    parser.add_argument("--test-size", type=int, default=21)
    parser.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help="Optional rolling training-window length; default uses all prior observations",
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    metrics = run_experiment(
        args.symbol,
        args.start,
        args.synthetic,
        args.output_dir,
        args.min_train_size,
        args.test_size,
        args.transaction_cost_bps,
        args.max_train_size,
    )
    print(metrics.round(4).to_string())


if __name__ == "__main__":
    main()
