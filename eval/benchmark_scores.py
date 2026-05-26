#!/usr/bin/env python3.12
"""Summarise Test 3 zero-shot OOD and Test 4 few-shot adaptation results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml


def _model_name(path: Path) -> str:
    name = path.name
    return name.replace("_test3.npz", "").replace("_test4_fewshot.npz", "")


def summarize_test3(path: Path, cfg: dict) -> dict:
    data = np.load(path)
    lengths = data["lengths"]
    masses = data["masses"]
    rewards = data["rewards"]

    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]

    L, M = np.meshgrid(lengths, masses)
    in_dist = (
        (L >= len_lo) & (L <= len_hi) &
        (M >= mass_lo) & (M <= mass_hi)
    )
    ood = ~in_dist

    id_mean = float(np.mean(rewards[in_dist]))
    ood_mean = float(np.mean(rewards[ood]))
    ood_p10 = float(np.quantile(rewards[ood], 0.10))
    retention = ood_mean / max(id_mean, 1e-6)
    return {
        "model": _model_name(path),
        "id_mean": id_mean,
        "ood_mean": ood_mean,
        "ood_p10": ood_p10,
        "retention": retention,
    }


def summarize_test4(path: Path, threshold: float) -> dict:
    data = np.load(path)
    budgets = data["budgets"].astype(float)
    rewards = data["rewards"].astype(float)
    mean_rewards = rewards.mean(axis=0)

    auc = float(np.trapezoid(mean_rewards, budgets) /
                max(1.0, budgets[-1] - budgets[0]))
    reached = budgets[mean_rewards >= threshold]
    episodes_to_threshold = float(reached[0]) if reached.size else np.nan
    return {
        "model": _model_name(path),
        "zero_shot": float(mean_rewards[0]),
        "final": float(mean_rewards[-1]),
        "gain": float(mean_rewards[-1] - mean_rewards[0]),
        "auc": auc,
        "episodes_to_threshold": episodes_to_threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarise benchmark result files.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--results_dir", default="eval/results")
    parser.add_argument("--threshold", type=float, default=1500.0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    result_dir = Path(args.results_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Missing results dir: {result_dir}")

    test3 = [summarize_test3(p, cfg) for p in sorted(result_dir.glob("*_test3.npz"))]
    test4 = [summarize_test4(p, args.threshold)
             for p in sorted(result_dir.glob("*_test4_fewshot.npz"))]

    if test3:
        print("\nTest 3: zero-shot OOD heatmap")
        print(f"{'model':32s} {'ID':>8s} {'OOD':>8s} {'p10':>8s} {'ret':>6s}")
        for row in sorted(test3, key=lambda r: r["ood_mean"], reverse=True):
            print(f"{row['model']:32s} {row['id_mean']:8.1f} "
                  f"{row['ood_mean']:8.1f} {row['ood_p10']:8.1f} "
                  f"{row['retention']:6.2f}")
    else:
        print("\nNo Test 3 result files found.")

    if test4:
        print("\nTest 4: few-shot OOD adaptation")
        print(f"{'model':32s} {'zero':>8s} {'final':>8s} {'gain':>8s} "
              f"{'auc':>8s} {'eps>=thr':>9s}")
        for row in sorted(test4, key=lambda r: r["auc"], reverse=True):
            eps = row["episodes_to_threshold"]
            eps_str = "nan" if np.isnan(eps) else f"{eps:.0f}"
            print(f"{row['model']:32s} {row['zero_shot']:8.1f} "
                  f"{row['final']:8.1f} {row['gain']:8.1f} "
                  f"{row['auc']:8.1f} {eps_str:>9s}")
    else:
        print("\nNo Test 4 result files found.")


if __name__ == "__main__":
    main()
