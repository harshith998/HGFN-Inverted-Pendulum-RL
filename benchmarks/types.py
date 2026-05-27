"""Generic benchmark interfaces for RL control OOD tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class TaskSpec:
    """One environment setting to evaluate."""

    name: str
    split: str
    severity: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    topology: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PolicyAdapter(Protocol):
    """Minimal policy wrapper expected by the benchmark runner."""

    def act(self, obs: Any, deterministic: bool = True) -> Any:
        ...


class EnvFactory(Protocol):
    """Build a fresh env for a task."""

    def __call__(self, task: TaskSpec) -> Any:
        ...


class PolicyFactory(Protocol):
    """Build a task-specific reference policy, e.g. oracle or random."""

    def __call__(self, task: TaskSpec) -> PolicyAdapter:
        ...


AdaptationFn = Callable[[PolicyAdapter, TaskSpec, int, EnvFactory], PolicyAdapter]

