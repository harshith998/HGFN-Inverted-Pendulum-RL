"""Few-shot OOD adaptation utilities for PPO-style continuous policies."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FewShotTask:
    name: str
    link_length: float
    link_mass: float


def compute_far_ood_tasks(cfg: dict, n_tasks: int = 4,
                          min_param_val: float = 0.05) -> list[FewShotTask]:
    """Pick deterministic far-OOD parameter points from the eval box corners."""
    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]

    len_w = len_hi - len_lo
    mass_w = mass_hi - mass_lo
    len_eval_lo = max(min_param_val, len_lo - len_w)
    len_eval_hi = len_hi + len_w
    mass_eval_lo = max(min_param_val, mass_lo - mass_w)
    mass_eval_hi = mass_hi + mass_w

    candidates = [
        FewShotTask("short_light", len_eval_lo, mass_eval_lo),
        FewShotTask("short_heavy", len_eval_lo, mass_eval_hi),
        FewShotTask("long_light", len_eval_hi, mass_eval_lo),
        FewShotTask("long_heavy", len_eval_hi, mass_eval_hi),
    ]
    return candidates[:max(1, min(n_tasks, len(candidates)))]


def obs_to_tensor(obs_list: list[dict], device: torch.device) -> dict:
    return {
        "node_features": torch.tensor(
            np.stack([o["node_features"] for o in obs_list]),
            dtype=torch.float32, device=device),
        "edge_index": torch.tensor(
            np.stack([o["edge_index"] for o in obs_list]),
            dtype=torch.int64, device=device),
        "edge_features": torch.tensor(
            np.stack([o["edge_features"] for o in obs_list]),
            dtype=torch.float32, device=device),
        "n_nodes": torch.tensor(
            np.stack([o["n_nodes"] for o in obs_list]),
            dtype=torch.int64, device=device),
        "n_edges": torch.tensor(
            np.stack([o["n_edges"] for o in obs_list]),
            dtype=torch.int64, device=device),
    }


def select_eval_action(policy, obs: dict, device: torch.device,
                       stochastic: bool = False) -> float:
    if not stochastic and hasattr(policy, "get_deterministic_action"):
        return float(policy.get_deterministic_action(obs, device))
    action, _, _ = policy.get_action(obs, device)
    return float(action)


def eval_continuous_policy(policy, make_env: Callable, cfg: dict,
                           task: FewShotTask, n_episodes: int,
                           device: torch.device,
                           stochastic: bool = False) -> float:
    env = make_env(cfg, task.link_length, task.link_mass)
    rewards = []
    for _ in range(n_episodes):
        try:
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action = select_eval_action(policy, obs, device, stochastic=stochastic)
                obs, reward, terminated, truncated, _ = env.step(
                    np.array([action], dtype=np.float32))
                ep_reward += reward
                done = terminated or truncated
            rewards.append(ep_reward)
        except Exception:
            rewards.append(0.0)
    env.close()
    return float(np.mean(rewards)) if rewards else 0.0


def collect_adaptation_episode(policy, env, device: torch.device,
                               gamma: float) -> dict:
    obs, _ = env.reset()
    done = False
    obs_list, actions, log_probs, values, rewards = [], [], [], [], []

    while not done:
        obs_t = obs_to_tensor([obs], device)
        with torch.no_grad():
            action_t, log_prob_t, _, value_t = policy.get_action_and_value(obs_t)

        action = float(action_t.squeeze())
        next_obs, reward, terminated, truncated, _ = env.step(
            np.array([action], dtype=np.float32))

        obs_list.append(obs)
        actions.append(action)
        log_probs.append(float(log_prob_t.squeeze()))
        values.append(float(value_t.squeeze()))
        rewards.append(float(reward))

        obs = next_obs
        done = terminated or truncated

    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()

    return {
        "obs": obs_list,
        "actions": np.array(actions, dtype=np.float32),
        "old_log_probs": np.array(log_probs, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "returns": np.array(returns, dtype=np.float32),
    }


def ppo_finetune(policy, episodes: list[dict], device: torch.device,
                 lr: float, epochs: int, batch_size: int,
                 clip_epsilon: float, value_coef: float,
                 entropy_coef: float, max_grad_norm: float):
    if not episodes:
        return

    obs_list = []
    actions, old_log_probs, returns, values = [], [], [], []
    for ep in episodes:
        obs_list.extend(ep["obs"])
        actions.append(ep["actions"])
        old_log_probs.append(ep["old_log_probs"])
        returns.append(ep["returns"])
        values.append(ep["values"])

    actions = torch.tensor(np.concatenate(actions), dtype=torch.float32,
                           device=device).unsqueeze(1)
    old_log_probs = torch.tensor(np.concatenate(old_log_probs), dtype=torch.float32,
                                 device=device)
    returns = torch.tensor(np.concatenate(returns), dtype=torch.float32,
                           device=device)
    old_values = torch.tensor(np.concatenate(values), dtype=torch.float32,
                              device=device)
    advantages = returns - old_values
    obs_t = obs_to_tensor(obs_list, device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = actions.shape[0]
    batch_size = max(1, min(batch_size, n))

    policy.train()
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            obs_b = {k: v[b] for k, v in obs_t.items()}
            adv_b = advantages[b]
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            _, new_log_probs, entropy, new_values = policy.get_action_and_value(
                obs_b, action=actions[b])
            ratio = (new_log_probs - old_log_probs[b]).exp()
            surr1 = ratio * adv_b
            surr2 = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values.squeeze(-1), returns[b])
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
    policy.eval()


def run_few_shot(policy, make_env: Callable, cfg: dict, device: torch.device,
                 budgets: list[int], n_tasks: int, eval_episodes: int,
                 adapt_lr: float, adapt_epochs: int, adapt_batch_size: int,
                 stochastic_eval: bool = False) -> tuple[list[FewShotTask], np.ndarray]:
    """Return rewards with shape (n_tasks, n_budgets)."""
    ppo_cfg = cfg["ppo"]
    budgets = sorted(set(int(b) for b in budgets if int(b) >= 0))
    if not budgets or budgets[0] != 0:
        budgets = [0] + budgets

    tasks = compute_far_ood_tasks(cfg, n_tasks=n_tasks)
    rewards = np.zeros((len(tasks), len(budgets)), dtype=np.float32)

    for task_idx, task in enumerate(tasks):
        adapted = copy.deepcopy(policy).to(device)
        adapted.eval()
        collected = []
        last_budget = 0

        print(f"\n[Test 4] task={task.name} "
              f"length={task.link_length:.3f} mass={task.link_mass:.3f}")

        for budget_idx, budget in enumerate(budgets):
            extra = budget - last_budget
            if extra > 0:
                env = make_env(cfg, task.link_length, task.link_mass)
                new_episodes = [
                    collect_adaptation_episode(adapted, env, device,
                                               gamma=ppo_cfg["gamma"])
                    for _ in range(extra)
                ]
                env.close()
                collected.extend(new_episodes)
                ppo_finetune(
                    adapted, new_episodes, device=device,
                    lr=adapt_lr, epochs=adapt_epochs,
                    batch_size=adapt_batch_size,
                    clip_epsilon=ppo_cfg["clip_epsilon"],
                    value_coef=ppo_cfg["value_coef"],
                    entropy_coef=ppo_cfg["entropy_coef"],
                    max_grad_norm=ppo_cfg["max_grad_norm"],
                )
                last_budget = budget

            reward = eval_continuous_policy(
                adapted, make_env, cfg, task, eval_episodes, device,
                stochastic=stochastic_eval,
            )
            rewards[task_idx, budget_idx] = reward
            print(f"  budget={budget:3d} eps | reward={reward:8.2f}")

    return tasks, rewards


def save_few_shot_results(path: str, tasks: list[FewShotTask],
                          budgets: list[int], rewards: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        task_names=np.array([t.name for t in tasks]),
        lengths=np.array([t.link_length for t in tasks], dtype=np.float64),
        masses=np.array([t.link_mass for t in tasks], dtype=np.float64),
        budgets=np.array(budgets, dtype=np.int64),
        rewards=rewards.astype(np.float32),
    )


def plot_few_shot(path: str, title: str, tasks: list[FewShotTask],
                  budgets: list[int], rewards: np.ndarray):
    fig, ax = plt.subplots(figsize=(9, 5))
    budgets_arr = np.array(budgets)
    for i, task in enumerate(tasks):
        ax.plot(budgets_arr, rewards[i], marker="o", linewidth=1.5,
                label=task.name)
    ax.plot(budgets_arr, rewards.mean(axis=0), color="black",
            linewidth=2.5, marker="o", label="mean")
    ax.set_xlabel("Fine-tuning episodes")
    ax.set_ylabel("Mean reward")
    ax.set_title(title)
    ax.set_ylim(0, 2000)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {path}")


def summarize_few_shot(budgets: list[int], rewards: np.ndarray,
                       threshold: float = 1500.0) -> dict:
    budgets_arr = np.array(budgets, dtype=np.float64)
    mean_rewards = rewards.mean(axis=0)
    auc = float(np.trapezoid(mean_rewards, budgets_arr) /
                max(1.0, budgets_arr[-1] - budgets_arr[0]))
    reached = budgets_arr[mean_rewards >= threshold]
    episodes_to_threshold = float(reached[0]) if reached.size else np.nan
    return {
        "zero_shot": float(mean_rewards[0]),
        "final": float(mean_rewards[-1]),
        "gain": float(mean_rewards[-1] - mean_rewards[0]),
        "auc": auc,
        "episodes_to_threshold": episodes_to_threshold,
    }
