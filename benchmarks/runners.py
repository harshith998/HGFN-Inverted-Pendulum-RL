"""Rollout helpers shared by all generic benchmarks."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import EnvFactory, PolicyAdapter, PolicyFactory, TaskSpec


def _reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, terminated or truncated, info
    obs, reward, done, info = out
    return obs, reward, done, info


def run_episode(env, policy: PolicyAdapter, deterministic: bool = True) -> float:
    """Run one episode against a Gym/Gymnasium-like env."""

    if hasattr(policy, "reset"):
        policy.reset()
    obs = _reset_env(env)
    total = 0.0
    done = False
    while not done:
        action = policy.act(obs, deterministic=deterministic)
        obs, reward, done, _ = _step_env(env, action)
        total += float(reward)
    return total


def evaluate_policy(
    env_factory: EnvFactory,
    policy: PolicyAdapter,
    tasks: Sequence[TaskSpec],
    n_episodes: int,
    deterministic: bool = True,
) -> dict[str, float]:
    """Evaluate one policy across tasks."""

    rewards = {}
    for task in tasks:
        vals = []
        for _ in range(n_episodes):
            env = env_factory(task)
            try:
                vals.append(run_episode(env, policy, deterministic=deterministic))
            finally:
                if hasattr(env, "close"):
                    env.close()
        rewards[task.name] = float(np.mean(vals))
    return rewards


def evaluate_reference(
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    tasks: Sequence[TaskSpec],
    n_episodes: int,
    deterministic: bool = True,
) -> dict[str, float]:
    """Evaluate task-specific reference policies such as oracle or random."""

    rewards = {}
    for task in tasks:
        policy = policy_factory(task)
        vals = []
        for _ in range(n_episodes):
            env = env_factory(task)
            try:
                vals.append(run_episode(env, policy, deterministic=deterministic))
            finally:
                if hasattr(env, "close"):
                    env.close()
        rewards[task.name] = float(np.mean(vals))
    return rewards

