"""Baseline random policy agent."""

from __future__ import annotations

from typing import Any

import numpy as np


class RandomAgent:
    """Uniform random action selection baseline.

    Useful as a sanity check: any learning agent should outperform this policy
    after sufficient training on most standard Gymnasium tasks.
    """

    def __init__(self, action_space, seed: int = 42) -> None:
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """No internal state to reset for a memoryless random policy."""

    def select_action(self, state: Any, training: bool = False) -> int:
        """Sample a random valid action."""
        if hasattr(self.action_space, "n"):
            return int(self.rng.integers(self.action_space.n))
        return int(self.action_space.sample())
