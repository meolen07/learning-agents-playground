"""Tabular Q-learning agent for discrete state/action MDPs."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


class QLearningAgent:
    """Classic off-policy temporal-difference control with an epsilon-greedy policy.

    Q-learning updates the action-value function with the one-step Bellman backup:
    Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    def reset(self) -> None:
        """Reset is a no-op; exploration schedule persists across episodes."""

    def _state_index(self, state: Any) -> int:
        if isinstance(state, (int, np.integer)):
            return int(state)
        return int(np.asarray(state).item())

    def select_action(self, state: Any, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        state_idx = self._state_index(state)
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state_idx]))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float:
        """Perform one Q-learning TD update and return the TD error."""
        s = self._state_index(state)
        ns = self._state_index(next_state)
        action = int(action)

        target = reward
        if not done:
            target += self.discount_factor * np.max(self.q_table[ns])

        td_error = target - self.q_table[s, action]
        self.q_table[s, action] += self.learning_rate * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Union[str, Path]) -> None:
        """Persist Q-table and hyperparameters to disk."""
        payload = {
            "q_table": self.q_table,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "QLearningAgent":
        """Load agent from a pickle checkpoint."""
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        agent = cls(
            n_states=payload["n_states"],
            n_actions=payload["n_actions"],
            learning_rate=payload["learning_rate"],
            discount_factor=payload["discount_factor"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
        )
        agent.q_table = payload["q_table"]
        return agent

    def get_q_values(self, state: Any) -> np.ndarray:
        """Return Q-values for all actions in a state."""
        return self.q_table[self._state_index(state)].copy()

    def policy(self, state: Any) -> int:
        """Greedy action without exploration."""
        return self.select_action(state, training=False)
