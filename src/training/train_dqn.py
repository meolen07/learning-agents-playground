"""DQN training loop for continuous-state Gymnasium environments."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from src.agents.dqn import DQNAgent
from src.envs.make_env import make_env
from src.evaluation.metrics import success_rate
from src.utils.plotting import plot_losses, plot_rewards
from src.utils.seeding import set_seed

logger = logging.getLogger(__name__)


def _flatten_obs(obs: Any) -> np.ndarray:
    """Convert observation to a 1-D float32 vector for the MLP."""
    return np.asarray(obs, dtype=np.float32).reshape(-1)


@dataclass
class DQNTrainResult:
    """Training artifacts returned by ``train_dqn``."""

    rewards: List[float]
    losses: List[float]
    agent: DQNAgent
    csv_path: Path
    plot_path: Path


def train_dqn(
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> DQNTrainResult:
    """Train a DQN agent from a configuration dictionary.

    Uses experience replay and a target network. Logs rewards and TD-loss,
    saves CSV logs, and writes training plots.
    """
    seed = int(config.get("seed", 42))
    env_id = config["env_id"]
    episodes = int(config.get("episodes", 500))
    max_steps = int(config.get("max_steps", 500))
    warmup_steps = int(config.get("warmup_steps", 1000))
    train_freq = int(config.get("train_freq", 4))
    success_threshold = float(config.get("success_threshold", 195.0))

    set_seed(seed)
    env, obs_space, action_space = make_env(env_id, seed=seed)

    state_dim = int(np.prod(obs_space.shape))
    action_dim = action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=float(config.get("learning_rate", 1e-3)),
        discount_factor=float(config.get("discount_factor", 0.99)),
        epsilon=float(config.get("epsilon", 1.0)),
        epsilon_min=float(config.get("epsilon_min", 0.01)),
        epsilon_decay=float(config.get("epsilon_decay", 0.995)),
        batch_size=int(config.get("batch_size", 64)),
        buffer_capacity=int(config.get("buffer_capacity", 100_000)),
        target_update_freq=int(config.get("target_update_freq", 10)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        seed=seed,
    )

    run_name = config.get("run_name", f"dqn_{env_id}")
    output_dir = Path(output_dir or config.get("output_dir", "results"))
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rewards: List[float] = []
    losses: List[float] = []
    global_step = 0
    csv_path = run_dir / "training_log.csv"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["episode", "reward", "success", "epsilon", "steps", "avg_loss"],
        )
        writer.writeheader()

        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=seed + episode)
            state = _flatten_obs(state)
            done = False
            total_reward = 0.0
            steps = 0
            episode_losses: List[float] = []

            while not done and steps < max_steps:
                action = agent.select_action(state, training=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = _flatten_obs(next_obs)
                done_flag = terminated or truncated

                agent.remember(state, action, float(reward), next_state, done_flag)
                state = next_state
                total_reward += float(reward)
                steps += 1
                global_step += 1

                if global_step > warmup_steps and global_step % train_freq == 0:
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(loss)
                        episode_losses.append(loss)

            agent.decay_epsilon()
            success = total_reward >= success_threshold
            rewards.append(total_reward)
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

            writer.writerow(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "success": int(success),
                    "epsilon": agent.epsilon,
                    "steps": steps,
                    "avg_loss": avg_loss,
                }
            )

            if verbose and episode % max(1, episodes // 10) == 0:
                logger.info(
                    "Episode %d/%d | reward=%.2f | avg_loss=%.4f | epsilon=%.3f",
                    episode,
                    episodes,
                    total_reward,
                    avg_loss,
                    agent.epsilon,
                )

    plot_window = int(config.get("plot_window", 50))
    plot_path = run_dir / "rewards.png"
    plot_rewards(rewards, window=plot_window, title=f"DQN on {env_id}", save_path=plot_path)
    if losses:
        plot_losses(
            losses,
            window=min(plot_window, len(losses)),
            title=f"DQN Loss on {env_id}",
            save_path=run_dir / "losses.png",
        )

    checkpoint_path = run_dir / "agent.pt"
    agent.save(checkpoint_path)
    env.close()

    return DQNTrainResult(
        rewards=rewards,
        losses=losses,
        agent=agent,
        csv_path=csv_path,
        plot_path=plot_path,
    )


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML training configuration."""
    with path.open("r") as handle:
        return yaml.safe_load(handle)
