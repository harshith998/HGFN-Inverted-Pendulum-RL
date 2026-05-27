"""Few-shot OOD adaptation benchmark entrypoints."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Sequence

from .metrics import few_shot_scores
from .runners import evaluate_policy, evaluate_reference
from .types import AdaptationFn, EnvFactory, PolicyAdapter, PolicyFactory, TaskSpec


def _clone_policy(policy: PolicyAdapter) -> PolicyAdapter:
    if hasattr(policy, "clone"):
        return policy.clone()
    return copy.deepcopy(policy)


def _tag(tasks: Sequence[TaskSpec], split: str) -> list[TaskSpec]:
    return [replace(t, split=split) for t in tasks]


def run_few_shot_benchmark(
    benchmark_name: str,
    env_factory: EnvFactory,
    base_policy: PolicyAdapter,
    random_policy_factory: PolicyFactory,
    oracle_policy_factory: PolicyFactory,
    adaptation_fn: AdaptationFn,
    tasks: Sequence[TaskSpec],
    budgets: Sequence[int],
    n_eval_episodes: int = 10,
    threshold: float = 0.8,
    deterministic: bool = True,
) -> dict:
    """Run static-budget few-shot tuning from the same base policy each time."""

    tasks = list(tasks)
    budgets = sorted(set([0] + list(budgets)))
    random_rewards = evaluate_reference(
        env_factory, random_policy_factory, tasks, n_eval_episodes, deterministic)
    oracle_rewards = evaluate_reference(
        env_factory, oracle_policy_factory, tasks, n_eval_episodes, deterministic)

    model_rewards: dict[str, list[float]] = {}
    for task in tasks:
        curve = []
        for budget in budgets:
            policy = _clone_policy(base_policy)
            if budget > 0:
                policy = adaptation_fn(policy, task, budget, env_factory)
            reward = evaluate_policy(
                env_factory, policy, [task], n_eval_episodes, deterministic)[task.name]
            curve.append(float(reward))
        model_rewards[task.name] = curve

    scores = few_shot_scores(
        tasks, budgets, model_rewards, random_rewards, oracle_rewards,
        threshold=threshold,
    )
    return {
        "benchmark": benchmark_name,
        "kind": "few_shot",
        "n_eval_episodes": n_eval_episodes,
        "scores": scores,
    }


def run_parameter_ood_few_shot(
    env_factory: EnvFactory,
    base_policy: PolicyAdapter,
    random_policy_factory: PolicyFactory,
    oracle_policy_factory: PolicyFactory,
    adaptation_fn: AdaptationFn,
    ood_tasks: Sequence[TaskSpec],
    budgets: Sequence[int],
    n_eval_episodes: int = 10,
    threshold: float = 0.8,
    deterministic: bool = True,
) -> dict:
    tasks = _tag(ood_tasks, "parameter_ood")
    return run_few_shot_benchmark(
        "parameter_ood_few_shot", env_factory, base_policy,
        random_policy_factory, oracle_policy_factory, adaptation_fn,
        tasks, budgets, n_eval_episodes, threshold, deterministic)


def run_topology_ood_few_shot(
    env_factory: EnvFactory,
    base_policy: PolicyAdapter,
    random_policy_factory: PolicyFactory,
    oracle_policy_factory: PolicyFactory,
    adaptation_fn: AdaptationFn,
    topology_ood_tasks: Sequence[TaskSpec],
    budgets: Sequence[int],
    n_eval_episodes: int = 10,
    threshold: float = 0.8,
    deterministic: bool = True,
) -> dict:
    tasks = _tag(topology_ood_tasks, "topology_ood")
    return run_few_shot_benchmark(
        "topology_ood_few_shot", env_factory, base_policy,
        random_policy_factory, oracle_policy_factory, adaptation_fn,
        tasks, budgets, n_eval_episodes, threshold, deterministic)
