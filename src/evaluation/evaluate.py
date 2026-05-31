"""Agent evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from src.evaluation.metrics import success_rate, summarize_rewards
from src.utils.seeding import set_seed


class AgentProtocol(Protocol):
    """Minimal agent interface for evaluation."""

    def select_action(self, state: Any, training: bool = False) -> int: ...

    def reset(self) -> None: ...


@dataclass
class EvaluationResult:
    """Container for evaluation outputs."""

    rewards: List[float]
    successes: List[bool]
    summary: Dict[str, float]

    @property
    def mean_reward(self) -> float:
        return self.summary["mean"]


def evaluate_agent(
    env,
    agent: AgentProtocol,
    episodes: int = 100,
    seed: int = 42,
    success_threshold: Optional[float] = None,
    render: bool = False,
) -> EvaluationResult:
    """Run a trained agent for a fixed number of evaluation episodes.

    Args:
        env: Gymnasium environment.
        agent: Agent implementing ``select_action`` and ``reset``.
        episodes: Number of evaluation rollouts.
        seed: Base seed; each episode uses ``seed + episode_index``.
        success_threshold: Reward threshold counting as success. When ``None``,
            success is defined as ``reward > 0`` for sparse-reward envs or
            ``reward >= env spec reward threshold`` when available.
        render: Whether to call ``env.render()`` each step.

    Returns:
        EvaluationResult with per-episode rewards and aggregate statistics.
    """
    set_seed(seed)
    rewards: List[float] = []
    successes: List[bool] = []

    if success_threshold is None:
        spec = getattr(env, "spec", None)
        if spec is not None and getattr(spec, "reward_threshold", None) is not None:
            success_threshold = float(spec.reward_threshold)
        else:
            success_threshold = 0.0

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        agent.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            if render:
                env.render()

        rewards.append(total_reward)
        successes.append(total_reward >= success_threshold)

    summary = summarize_rewards(rewards)
    summary["success_rate"] = success_rate(successes)

    return EvaluationResult(rewards=rewards, successes=successes, summary=summary)
