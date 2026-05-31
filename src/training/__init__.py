"""Training routines."""

from src.training.train_q_learning import (
    QLearningTrainResult,
    load_config as load_qlearning_config,
    train_q_learning,
)

__all__ = [
    "train_q_learning",
    "train_dqn",
    "QLearningTrainResult",
    "DQNTrainResult",
    "load_qlearning_config",
    "load_dqn_config",
]


def __getattr__(name: str):
    """Lazy-load DQN training to avoid importing torch at package import time."""
    if name in {"train_dqn", "DQNTrainResult", "load_dqn_config"}:
        from src.training.train_dqn import DQNTrainResult, load_config, train_dqn

        if name == "train_dqn":
            return train_dqn
        if name == "DQNTrainResult":
            return DQNTrainResult
        return load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
