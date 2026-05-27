"""Oracle-normalized benchmark scores."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .types import TaskSpec


def normalized_score(model, random, oracle, eps: float = 1e-8, clip: bool = True):
    """Map raw reward to 0=random baseline, 1=oracle baseline."""

    model = np.asarray(model, dtype=float)
    random = np.asarray(random, dtype=float)
    oracle = np.asarray(oracle, dtype=float)
    score = (model - random) / np.maximum(np.abs(oracle - random), eps)
    return np.clip(score, 0.0, 1.0) if clip else score


def zero_shot_scores(
    tasks: Sequence[TaskSpec],
    model_rewards: dict[str, float],
    random_rewards: dict[str, float],
    oracle_rewards: dict[str, float],
) -> dict:
    """Summarize ID/OOD performance with oracle-normalized scores."""

    by_split: dict[str, list[float]] = defaultdict(list)
    per_task = {}
    for task in tasks:
        score = float(normalized_score(
            model_rewards[task.name],
            random_rewards[task.name],
            oracle_rewards[task.name],
        ))
        by_split[task.split].append(score)
        per_task[task.name] = {
            "split": task.split,
            "severity": task.severity,
            "score": score,
            "model_reward": float(model_rewards[task.name]),
            "random_reward": float(random_rewards[task.name]),
            "oracle_reward": float(oracle_rewards[task.name]),
        }

    ood_vals = [
        v for split, vals in by_split.items()
        if split != "id"
        for v in vals
    ]
    id_vals = by_split.get("id", [])
    id_mean = float(np.mean(id_vals)) if id_vals else float("nan")
    ood_mean = float(np.mean(ood_vals)) if ood_vals else float("nan")
    ood_p10 = float(np.quantile(ood_vals, 0.10)) if ood_vals else float("nan")
    retention = float(ood_mean / max(id_mean, 1e-8)) if id_vals and ood_vals else float("nan")

    robustness = np.clip(retention, 0.0, 1.0) if np.isfinite(retention) else 0.0
    tail = np.clip(ood_p10, 0.0, 1.0) if np.isfinite(ood_p10) else 0.0
    ood = np.clip(ood_mean, 0.0, 1.0) if np.isfinite(ood_mean) else 0.0
    composite = float(0.50 * ood + 0.25 * robustness + 0.25 * tail)

    return {
        "id_score": id_mean,
        "ood_score": ood_mean,
        "ood_p10": ood_p10,
        "retention": retention,
        "composite": composite,
        "per_task": per_task,
    }


def few_shot_scores(
    tasks: Sequence[TaskSpec],
    budgets: Sequence[int],
    model_rewards: dict[str, list[float]],
    random_rewards: dict[str, float],
    oracle_rewards: dict[str, float],
    threshold: float = 0.8,
) -> dict:
    """Summarize adaptation efficiency on normalized reward curves."""

    budgets_arr = np.asarray(sorted(set([0] + list(budgets))), dtype=float)
    curves = []
    per_task = {}
    for task in tasks:
        raw = np.asarray(model_rewards[task.name], dtype=float)
        norm = normalized_score(
            raw,
            random_rewards[task.name],
            oracle_rewards[task.name],
        )
        curves.append(norm)
        reached = budgets_arr[norm >= threshold]
        per_task[task.name] = {
            "split": task.split,
            "severity": task.severity,
            "normalized_curve": norm.tolist(),
            "raw_rewards": raw.tolist(),
            "episodes_to_threshold": (
                float(reached[0]) if reached.size else float("nan")
            ),
        }

    mean_curve = np.mean(np.vstack(curves), axis=0)
    denom = max(1.0, float(budgets_arr[-1] - budgets_arr[0]))
    auc = float(np.trapezoid(mean_curve, budgets_arr) / denom)
    reached = budgets_arr[mean_curve >= threshold]
    episodes_to_threshold = float(reached[0]) if reached.size else float("nan")
    speed = 0.0 if not reached.size else 1.0 - episodes_to_threshold / max(1.0, budgets_arr[-1])
    gain = float(mean_curve[-1] - mean_curve[0])
    composite = float(
        0.35 * mean_curve[-1] +
        0.25 * auc +
        0.20 * np.clip(speed, 0.0, 1.0) +
        0.20 * np.clip(gain, 0.0, 1.0)
    )

    return {
        "zero_shot": float(mean_curve[0]),
        "final": float(mean_curve[-1]),
        "gain": gain,
        "auc": auc,
        "episodes_to_threshold": episodes_to_threshold,
        "threshold": float(threshold),
        "composite": composite,
        "budgets": budgets_arr.astype(int).tolist(),
        "mean_curve": mean_curve.tolist(),
        "per_task": per_task,
    }

