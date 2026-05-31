"""Lightweight smoke tests for Learning Agents Playground."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.q_learning import QLearningAgent
from src.agents.random_agent import RandomAgent
from src.envs.make_env import make_env
from src.evaluation.evaluate import evaluate_agent
from src.evaluation.metrics import moving_average, success_rate, summarize_rewards
from src.training.train_q_learning import train_q_learning
from src.utils.replay_buffer import ReplayBuffer
from src.utils.seeding import set_seed


def _torch_available() -> bool:
    """Return True when torch is installed and torch tests are not explicitly skipped."""
    if os.environ.get("LAP_SKIP_TORCH_TESTS", "").lower() in {"1", "true", "yes"}:
        return False
    return importlib.util.find_spec("torch") is not None


TORCH_AVAILABLE = _torch_available()
requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch is unavailable in this environment",
)


def test_set_seed_reproducibility() -> None:
    set_seed(123)
    a = np.random.rand()
    set_seed(123)
    b = np.random.rand()
    assert a == b


def test_make_env_frozenlake() -> None:
    env, obs_space, action_space = make_env("FrozenLake-v1", seed=42, is_slippery=False)
    assert obs_space.n == 16
    assert action_space.n == 4
    state, _ = env.reset()
    assert 0 <= state < 16
    env.close()


def test_random_agent_action() -> None:
    env, _, action_space = make_env("FrozenLake-v1", seed=1, is_slippery=False)
    agent = RandomAgent(action_space, seed=1)
    action = agent.select_action(0)
    assert 0 <= action < action_space.n
    env.close()


def test_q_learning_update_and_save(tmp_path: Path) -> None:
    agent = QLearningAgent(n_states=4, n_actions=2, seed=0)
    td_error = agent.update(0, 0, 1.0, 1, False)
    assert isinstance(td_error, float)

    checkpoint = tmp_path / "q.pkl"
    agent.save(checkpoint)
    loaded = QLearningAgent.load(checkpoint)
    assert np.allclose(loaded.q_table, agent.q_table)


def test_replay_buffer_sample() -> None:
    buffer = ReplayBuffer(capacity=10)
    for i in range(5):
        state = np.array([float(i)], dtype=np.float32)
        buffer.push(state, 0, 1.0, state, False)
    batch = buffer.sample(3)
    assert len(batch) == 5
    assert batch[0].shape[0] == 3


@requires_torch
def test_qnetwork_forward() -> None:
    import torch

    from src.agents.dqn import QNetwork

    net = QNetwork(state_dim=4, action_dim=2, hidden_dim=128)
    out = net(torch.zeros(1, 4))
    assert out.shape == (1, 2)


@requires_torch
def test_dqn_train_step() -> None:
    from src.agents.dqn import DQNAgent

    agent = DQNAgent(state_dim=4, action_dim=2, batch_size=4, seed=0)
    for _ in range(8):
        state = np.random.rand(4).astype(np.float32)
        next_state = np.random.rand(4).astype(np.float32)
        agent.remember(state, 0, 1.0, next_state, False)
    loss = agent.train_step()
    assert loss is not None


def test_metrics_helpers() -> None:
    rewards = [1.0, 2.0, 3.0]
    summary = summarize_rewards(rewards)
    assert summary["mean"] == 2.0
    assert moving_average(rewards, window=2)[-1] == 2.5
    assert success_rate([True, False, True]) == pytest.approx(2 / 3)


def test_short_q_learning_training(tmp_path: Path) -> None:
    config = {
        "env_id": "FrozenLake-v1",
        "episodes": 5,
        "seed": 0,
        "is_slippery": False,
        "run_name": "smoke_qlearning",
        "output_dir": str(tmp_path),
    }
    result = train_q_learning(config, verbose=False)
    assert len(result.rewards) == 5
    assert result.csv_path.exists()
    assert result.plot_path.exists()


@requires_torch
def test_short_dqn_training(tmp_path: Path) -> None:
    from src.training.train_dqn import train_dqn

    config = {
        "env_id": "CartPole-v1",
        "episodes": 2,
        "seed": 0,
        "warmup_steps": 10,
        "run_name": "smoke_dqn",
        "output_dir": str(tmp_path),
        "max_steps": 20,
    }
    result = train_dqn(config, verbose=False)
    assert len(result.rewards) == 2
    assert result.csv_path.exists()


def test_evaluate_q_learning_agent() -> None:
    env, obs_space, action_space = make_env("FrozenLake-v1", seed=0, is_slippery=False)
    agent = QLearningAgent(n_states=obs_space.n, n_actions=action_space.n, seed=0)
    result = evaluate_agent(env, agent, episodes=3, seed=0)
    assert len(result.rewards) == 3
    assert "success_rate" in result.summary
    env.close()
