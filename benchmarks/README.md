# Unified RL Control OOD Benchmarks

This folder is intentionally project-agnostic. It does not know about pendulums, ants, graph observations, PPO, or LQR.

You provide:

- `TaskSpec` objects: ID, parameter-OOD, or topology-OOD settings.
- `env_factory(task)`: builds a fresh env for that task.
- `model_policy.act(obs, deterministic=True)`: your controller.
- `random_policy_factory(task)`: weak reference controller.
- `oracle_policy_factory(task)`: task-aware reference controller.
- `adaptation_fn(policy, task, budget, env_factory)`: only for few-shot benchmarks.

Scores are normalized per task:

```text
score = (model_reward - random_reward) / (oracle_reward - random_reward)
```

So `0` means random-level and `1` means oracle-level. Few-shot threshold defaults to `0.8`, meaning 80% of the oracle gap.

The four intended benchmarks are:

- `run_parameter_ood_zero_shot`
- `run_topology_ood_zero_shot`
- `run_parameter_ood_few_shot`
- `run_topology_ood_few_shot`

Topology mutation is not hardcoded here. The benchmark owns severity and scoring; the robotics project owns what “add one joint,” “add one leg,” or “rewire one edge” means for its simulator.

