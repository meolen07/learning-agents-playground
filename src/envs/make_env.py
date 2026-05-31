"""Environment factory with reproducible seeding."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import gymnasium as gym

from src.utils.seeding import set_seed


def make_env(
    env_id: str,
    seed: int = 42,
    render_mode: Optional[str] = None,
    is_slippery: bool = True,
    **kwargs: Any,
) -> Tuple[gym.Env, gym.spaces.Space, gym.spaces.Space]:
    """Create a Gymnasium environment with synchronized random seeds.

    For ``FrozenLake-v1``, the ``is_slippery`` flag controls whether transitions
    are deterministic (False) or stochastic (True), which strongly affects how
    difficult tabular Q-learning is to train.

    Args:
        env_id: Registered Gymnasium environment identifier.
        seed: Random seed for reproducibility.
        render_mode: Optional render mode passed to ``gym.make``.
        is_slippery: Whether FrozenLake tiles are slippery (ignored for other envs).
        **kwargs: Additional keyword arguments forwarded to ``gym.make``.

    Returns:
        Tuple of (environment, observation_space, action_space).
    """
    make_kwargs = dict(kwargs)
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode

    if env_id.startswith("FrozenLake"):
        make_kwargs.setdefault("is_slippery", is_slippery)

    env = gym.make(env_id, **make_kwargs)
    set_seed(seed, env=env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env, env.observation_space, env.action_space
