"""RL agent implementations."""

from src.agents.q_learning import QLearningAgent
from src.agents.random_agent import RandomAgent

__all__ = [
    "RandomAgent",
    "QLearningAgent",
    "DQNAgent",
    "QNetwork",
]


def __getattr__(name: str):
    """Lazy-load torch-dependent agents to keep import side effects minimal."""
    if name in {"DQNAgent", "QNetwork"}:
        from src.agents.dqn import DQNAgent, QNetwork

        return DQNAgent if name == "DQNAgent" else QNetwork
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
