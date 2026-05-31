#!/usr/bin/env python3
"""CLI entry point for agent evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.dqn import DQNAgent
from src.agents.q_learning import QLearningAgent
from src.envs.make_env import make_env
from src.evaluation.evaluate import evaluate_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent.")
    parser.add_argument(
        "--agent-type",
        choices=["qlearning", "dqn"],
        required=True,
        help="Type of agent checkpoint to load.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to agent checkpoint (.pkl or .pt).",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        required=True,
        help="Gymnasium environment ID.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--is-slippery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FrozenLake slippery flag (ignored for other envs).",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="Reward threshold for counting success.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file for evaluation summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env, _, _ = make_env(args.env_id, seed=args.seed, is_slippery=args.is_slippery)

    if args.agent_type == "qlearning":
        agent = QLearningAgent.load(args.checkpoint)
    else:
        agent = DQNAgent.load(args.checkpoint)

    result = evaluate_agent(
        env,
        agent,
        episodes=args.episodes,
        seed=args.seed,
        success_threshold=args.success_threshold,
    )
    env.close()

    print(json.dumps(result.summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(result.summary, handle, indent=2)


if __name__ == "__main__":
    main()
