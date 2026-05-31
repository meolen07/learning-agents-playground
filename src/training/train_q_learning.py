"""Q-learning training loop for discrete tabular environments."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.agents.q_learning import QLearningAgent
from src.envs.make_env import make_env
from src.evaluation.metrics import success_rate
from src.utils.plotting import plot_rewards, plot_success_rate
from src.utils.seeding import set_seed

logger = logging.getLogger(__name__)


@dataclass
class QLearningTrainResult:
    """Training artifacts returned by ``train_q_learning``."""

    rewards: List[float]
    successes: List[bool]
    agent: QLearningAgent
    csv_path: Path
    plot_path: Path


def train_q_learning(
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> QLearningTrainResult:
    """Train a tabular Q-learning agent from a configuration dictionary.

    Logs episode rewards, saves CSV logs, and writes reward/success plots.
    """
    seed = int(config.get("seed", 42))
    env_id = config["env_id"]
    episodes = int(config.get("episodes", 5000))
    is_slippery = bool(config.get("is_slippery", True))
    success_threshold = float(config.get("success_threshold", 0.0))

    set_seed(seed)
    env, obs_space, action_space = make_env(
        env_id,
        seed=seed,
        is_slippery=is_slippery,
    )

    n_states = obs_space.n
    n_actions = action_space.n

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=float(config.get("learning_rate", 0.1)),
        discount_factor=float(config.get("discount_factor", 0.99)),
        epsilon=float(config.get("epsilon", 1.0)),
        epsilon_min=float(config.get("epsilon_min", 0.05)),
        epsilon_decay=float(config.get("epsilon_decay", 0.995)),
        seed=seed,
    )

    run_name = config.get("run_name", f"qlearning_{env_id}")
    output_dir = Path(output_dir or config.get("output_dir", "results"))
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rewards: List[float] = []
    successes: List[bool] = []
    csv_path = run_dir / "training_log.csv"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["episode", "reward", "success", "epsilon", "steps"],
        )
        writer.writeheader()

        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.update(state, action, reward, next_state, terminated or truncated)
                state = next_state
                total_reward += float(reward)
                done = terminated or truncated
                steps += 1

            agent.decay_epsilon()
            success = total_reward >= success_threshold
            rewards.append(total_reward)
            successes.append(success)

            writer.writerow(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "success": int(success),
                    "epsilon": agent.epsilon,
                    "steps": steps,
                }
            )

            if verbose and episode % max(1, episodes // 10) == 0:
                logger.info(
                    "Episode %d/%d | reward=%.2f | success_rate=%.2f | epsilon=%.3f",
                    episode,
                    episodes,
                    total_reward,
                    success_rate(successes[-100:]),
                    agent.epsilon,
                )

    plot_window = int(config.get("plot_window", 50))
    plot_path = run_dir / "rewards.png"
    plot_rewards(rewards, window=plot_window, title=f"Q-Learning on {env_id}", save_path=plot_path)
    plot_success_rate(
        successes,
        window=plot_window,
        title=f"Success Rate on {env_id}",
        save_path=run_dir / "success_rate.png",
    )

    checkpoint_path = run_dir / "agent.pkl"
    agent.save(checkpoint_path)
    env.close()

    return QLearningTrainResult(
        rewards=rewards,
        successes=successes,
        agent=agent,
        csv_path=csv_path,
        plot_path=plot_path,
    )


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML training configuration."""
    with path.open("r") as handle:
        return yaml.safe_load(handle)
