"""Visualization helpers for training and evaluation logs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import moving_average


PathLike = Union[str, Path]


def plot_rewards(
    rewards: Sequence[float],
    window: int = 50,
    title: str = "Episode Rewards",
    save_path: Optional[PathLike] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot raw episode rewards with a moving-average overlay."""
    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = np.arange(1, len(rewards) + 1)
    ax.plot(episodes, rewards, alpha=0.35, label="Episode reward")
    if len(rewards) >= window:
        smoothed = moving_average(rewards, window)
        ax.plot(episodes, smoothed, linewidth=2, label=f"{window}-episode moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_success_rate(
    successes: Sequence[bool],
    window: int = 50,
    title: str = "Success Rate",
    save_path: Optional[PathLike] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot rolling success rate from boolean episode outcomes."""
    success_floats = [1.0 if s else 0.0 for s in successes]
    rolling = moving_average(success_floats, window)

    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = np.arange(1, len(successes) + 1)
    ax.plot(episodes, rolling, linewidth=2, label=f"{window}-episode success rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_losses(
    losses: Sequence[float],
    window: int = 50,
    title: str = "Training Loss",
    save_path: Optional[PathLike] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot DQN TD-loss curve with optional smoothing."""
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = np.arange(1, len(losses) + 1)
    ax.plot(steps, losses, alpha=0.35, label="Step loss")
    if len(losses) >= window:
        smoothed = moving_average(losses, window)
        ax.plot(steps, smoothed, linewidth=2, label=f"{window}-step moving avg")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_from_csv(
    csv_path: PathLike,
    reward_column: str = "reward",
    window: int = 50,
    save_path: Optional[PathLike] = None,
    show: bool = False,
) -> plt.Figure:
    """Load a training CSV and plot the reward column."""
    df = pd.read_csv(csv_path)
    if reward_column not in df.columns:
        raise KeyError(f"Column '{reward_column}' not found in {csv_path}")
    return plot_rewards(
        df[reward_column].tolist(),
        window=window,
        title=f"Rewards from {Path(csv_path).name}",
        save_path=save_path,
        show=show,
    )


def plot_comparison(
    series: Iterable[tuple[str, Sequence[float]]],
    window: int = 50,
    title: str = "Reward Comparison",
    save_path: Optional[PathLike] = None,
    show: bool = False,
) -> plt.Figure:
    """Overlay multiple reward series on one chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, rewards in series:
        episodes = np.arange(1, len(rewards) + 1)
        if len(rewards) >= window:
            smoothed = moving_average(rewards, window)
            ax.plot(episodes, smoothed, linewidth=2, label=label)
        else:
            ax.plot(episodes, rewards, linewidth=2, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
