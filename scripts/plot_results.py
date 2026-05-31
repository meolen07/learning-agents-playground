#!/usr/bin/env python3
"""CLI utility to plot training CSV logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plotting import plot_from_csv, plot_rewards, plot_success_rate
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training results from CSV logs.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to training_log.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (defaults to CSV parent directory).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Moving-average window size.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_from_csv(
        args.csv,
        window=args.window,
        save_path=output_dir / "replotted_rewards.png",
        show=args.show,
    )

    df = pd.read_csv(args.csv)
    if "success" in df.columns:
        plot_success_rate(
            df["success"].astype(bool).tolist(),
            window=args.window,
            save_path=output_dir / "replotted_success_rate.png",
            show=args.show,
        )

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
