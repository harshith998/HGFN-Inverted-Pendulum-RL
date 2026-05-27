"""Zero-shot OOD benchmark entrypoints."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .metrics import zero_shot_scores
from .runners import evaluate_policy, evaluate_reference
from .types import EnvFactory, PolicyAdapter, PolicyFactory, TaskSpec


def _tag(tasks: Sequence[TaskSpec], split: str) -> list[TaskSpec]:
    return [replace(t, split=split) for t in tasks]


def run_zero_shot_benchmark(
    benchmark_name: str,
    env_factory: EnvFactory,
    model_policy: PolicyAdapter,
    random_policy_factory: PolicyFactory,
    oracle_policy_factory: PolicyFactory,
    tasks: Sequence[TaskSpec],
    n_episodes: int = 10,
    deterministic: bool = True,
) -> dict:
    """Run a generic oracle-normalized zero-shot benchmark."""

    tasks = list(tasks)
    model_rewards = evaluate_policy(
        env_factory, model_policy, tasks, n_episodes, deterministic)
    random_rewards = evaluate_reference(
        env_factory, random_policy_factory, tasks, n_episodes, deterministic)
    oracle_rewards = evaluate_reference(
        env_factory, oracle_policy_factory, tasks, n_episodes, deterministic)
    scores = zero_shot_scores(tasks, model_rewards, random_rewards, oracle_rewards)
    return {
        "benchmark": benchmark_name,
        "kind": "zero_shot",
        "n_episodes": n_episodes,
        "scores": scores,
    }


def run_parameter_ood_zero_shot(
    env_factory: EnvFactory,
    model_policy: PolicyAdapter,
    random_policy_factory: PolicyFactory,
    oracle_policy_factory: PolicyFactory,
    id_tasks: Sequence[TaskSpec],
    near_ood_tasks: Sequence[TaskSpec],
    far_ood_tasks: Sequence[TaskSpec],
    n_episodes: int = 10,
    deterministic: bool = True,
) -> dict:
    tasks = _tag(id_tasks, "id") + _tag(near_ood_tasks, "near_ood") + _tag(far_ood_tasks, "far_ood")
    return run_zero_shot_benchmark(
        "parameter_ood_zero_shot", env_factory, model_policy,
        random_policy_factory, oracle_policy_factory, tasks,
        n_episodes=n_episodes, deterministic=deterministic)


def run_topology_ood_zero_shot(
    env_factory: EnvFactory,
    model_policy: PolicyAdapter,
    random_policy_factory: PolicyFactory,
    oracle_policy_factory: PolicyFactory,
    id_tasks: Sequence[TaskSpec],
    topology_ood_tasks: Sequence[TaskSpec],
    n_episodes: int = 10,
    deterministic: bool = True,
) -> dict:
    tasks = _tag(id_tasks, "id") + _tag(topology_ood_tasks, "topology_ood")
    return run_zero_shot_benchmark(
        "topology_ood_zero_shot", env_factory, model_policy,
        random_policy_factory, oracle_policy_factory, tasks,
        n_episodes=n_episodes, deterministic=deterministic)

