"""Reproducibility utilities for RL experiments."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, env: Optional[object] = None) -> None:
    """Set random seeds for Python, NumPy, and optionally a Gymnasium environment.

    Reproducible RL runs require synchronized RNG state across libraries so that
    environment transitions, agent exploration, and weight initialization match
    across repeated experiments.

    Args:
        seed: Integer seed applied to all supported RNGs.
        env: Optional Gymnasium environment; ``reset(seed=...)`` is called when
            the environment supports seeding.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if env is not None and hasattr(env, "reset"):
        try:
            env.reset(seed=seed)
        except TypeError:
            # Older Gym API without seed kwarg on reset.
            if hasattr(env, "seed"):
                env.seed(seed)
