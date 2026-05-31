#!/usr/bin/env python3
"""CLI entry point for DQN training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_dqn import load_config, train_dqn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cartpole_dqn.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of training episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.episodes is not None:
        config["episodes"] = args.episodes
    if args.seed is not None:
        config["seed"] = args.seed

    result = train_dqn(config, verbose=args.verbose)
    print(f"Training complete. Logs saved to {result.csv_path}")


if __name__ == "__main__":
    main()
