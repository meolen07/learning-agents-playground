"""Evaluation metrics for RL agents."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def moving_average(values: Sequence[float], window: int) -> List[float]:
    """Compute a simple moving average with a trailing window.

    Early indices use all available history up to ``window`` samples, which
    avoids dropping the first part of a learning curve when plotting.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if not values:
        return []

    result: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def summarize_rewards(rewards: Sequence[float]) -> Dict[str, float]:
    """Return basic descriptive statistics for a reward sequence."""
    if not rewards:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }

    count = len(rewards)
    mean = sum(rewards) / count
    variance = sum((r - mean) ** 2 for r in rewards) / count
    return {
        "count": float(count),
        "mean": mean,
        "min": float(min(rewards)),
        "max": float(max(rewards)),
        "std": float(variance**0.5),
    }


def success_rate(successes: Iterable[bool]) -> float:
    """Fraction of episodes marked successful."""
    success_list = list(successes)
    if not success_list:
        return 0.0
    return sum(1 for s in success_list if s) / len(success_list)
