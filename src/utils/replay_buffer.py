"""Experience replay buffer for off-policy deep RL."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np


@dataclass
class Transition:
    """Single environment transition stored for replay."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity FIFO buffer for DQN experience replay.

    Breaking temporal correlation in mini-batches stabilizes Q-learning when
    training from a neural network approximator.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition."""
        self._buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Uniformly sample a batch of transitions."""
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from buffer of size {len(self._buffer)}"
            )

        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]

        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def clear(self) -> None:
        """Remove all stored transitions."""
        self._buffer.clear()
