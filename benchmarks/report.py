"""Small reporting helpers for benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def save_json(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(result), indent=2) + "\n")


def compact_summary(result: dict) -> dict:
    scores = result["scores"]
    keys = [
        "id_score", "ood_score", "ood_p10", "retention",
        "zero_shot", "final", "gain", "auc",
        "episodes_to_threshold", "composite",
    ]
    return {
        "benchmark": result.get("benchmark"),
        "kind": result.get("kind"),
        **{k: scores[k] for k in keys if k in scores},
    }

