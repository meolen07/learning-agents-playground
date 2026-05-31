"""Deep Q-Network (DQN) agent with target network and experience replay."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """Two-hidden-layer MLP Q-function approximator (128-128)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DQNAgent:
    """DQN with epsilon-greedy exploration and periodic target-network sync.

    The target network stabilizes bootstrapped Q-targets by keeping a slowly
    moving copy of the online network parameters.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 10,
        hidden_dim: int = 128,
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.rng = np.random.default_rng(seed)
        self.train_steps = 0

    def reset(self) -> None:
        """No episode-local state for DQN."""

    def _to_tensor(self, state: Any) -> torch.Tensor:
        array = np.asarray(state, dtype=np.float32).reshape(1, -1)
        return torch.from_numpy(array).to(self.device)

    def select_action(self, state: Any, training: bool = True) -> int:
        """Epsilon-greedy action from the online Q-network."""
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_dim))

        with torch.no_grad():
            q_values = self.q_network(self._to_tensor(state))
        return int(q_values.argmax(dim=1).item())

    def remember(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """Sample a mini-batch and perform one gradient descent step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Current Q(s,a)
        q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # max_a' Q_target(s', a') — standard DQN target (no Double DQN here)
            next_q = self.target_network(next_states_t).max(dim=1).values
            targets = rewards_t + self.discount_factor * next_q * (1.0 - dones_t)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Hard-copy online weights into the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Union[str, Path]) -> None:
        """Save online network weights and training metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_steps": self.train_steps,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "batch_size": self.batch_size,
                "target_update_freq": self.target_update_freq,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "DQNAgent":
        """Load agent checkpoint from disk."""
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        agent = cls(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            learning_rate=checkpoint["learning_rate"],
            discount_factor=checkpoint["discount_factor"],
            epsilon=checkpoint["epsilon"],
            epsilon_min=checkpoint["epsilon_min"],
            epsilon_decay=checkpoint["epsilon_decay"],
            batch_size=checkpoint["batch_size"],
            target_update_freq=checkpoint["target_update_freq"],
            device=device,
        )
        agent.q_network.load_state_dict(checkpoint["q_network"])
        agent.target_network.load_state_dict(checkpoint["target_network"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.train_steps = checkpoint.get("train_steps", 0)
        return agent

    def q_values(self, state: Any) -> np.ndarray:
        """Return Q-values for all actions."""
        with torch.no_grad():
            values = self.q_network(self._to_tensor(state)).cpu().numpy().reshape(-1)
        return values
