# evaluate_hierarchical.py
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from env.farm_env import FarmEnv
from env.hierarchical.lane_executor_env import LaneExecutorEnv
from env.hierarchical.high_level_env import HighLevelFarmEnv


# ── Step 1 baseline ───────────────────────────────────────────────────

def run_flat_episode(model, env, seed):
    obs, _ = env.reset(seed=seed)
    total_r, steps = 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, r, terminated, truncated, info = env.step(int(action))
        total_r += r
        steps += 1
    return total_r, steps, info["coverage"], terminated


def evaluate_flat(model_path="models/farm_ppo", n_episodes=50):
    env = FarmEnv(n_beds=4, field_height=8)
    model = MaskablePPO.load(model_path)
    results = [run_flat_episode(model, env, ep) for ep in range(n_episodes)]
    rewards, steps, covs, succs = zip(*results)
    print(f"\n[Step 1 - Flat PPO]  n={n_episodes}")
    print(f"  Success rate:  {np.mean(succs):.1%}")
    print(f"  Avg reward:    {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Avg coverage:  {np.mean(covs):.1%}")
    print(f"  Avg steps:     {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    return {"rewards": list(rewards), "steps": list(steps),
            "covs": list(covs), "succs": list(succs), "label": "Step1 Flat PPO"}


# ── Step 2 hierarchical ───────────────────────────────────────────────

def run_hl_episode(hl_model, ll_model, seed):
    env = HighLevelFarmEnv(ll_model, n_beds=4, field_height=8)
    obs, _ = env.reset(seed=seed)
    total_r, total_steps, visits = 0.0, 0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = hl_model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(int(action))
        total_r += r
        total_steps += info["steps_for_lane"]
        visits += 1
    return total_r, total_steps, info["coverage"], terminated, visits


def evaluate_hierarchical(
    ll_path="models/lane_executor",
    hl_path="models/high_level",
    n_episodes=50,
):
    ll_model = MaskablePPO.load(ll_path)
    hl_model = PPO.load(hl_path)
    results = [run_hl_episode(hl_model, ll_model, ep) for ep in range(n_episodes)]
    rewards, steps, covs, succs, visits = zip(*results)
    print(f"\n[Step 2 - Hierarchical RL]  n={n_episodes}")
    print(f"  Success rate:  {np.mean(succs):.1%}")
    print(f"  Avg reward:    {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Avg coverage:  {np.mean(covs):.1%}")
    print(f"  Avg steps:     {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"  Avg lane visits: {np.mean(visits):.1f}")
    return {"rewards": list(rewards), "steps": list(steps),
            "covs": list(covs), "succs": list(succs), "label": "Step2 Hierarchical"}


# ── Comparison plot ───────────────────────────────────────────────────

def plot_comparison(flat, hier):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Step 1 (Flat PPO) vs Step 2 (Hierarchical RL)", fontsize=13, fontweight="bold")

    for ax, key, xlabel, title in [
        (axes[0], "steps",   "Steps",            "Completion Steps"),
        (axes[1], "covs",    "Coverage Rate",     "Field Coverage"),
        (axes[2], "rewards", "Cumulative Reward",  "Episode Reward"),
    ]:
        ax.hist(flat[key],  bins=15, alpha=0.6, color="steelblue",
                label=flat["label"],  edgecolor="white")
        ax.hist(hier[key],  bins=15, alpha=0.6, color="coral",
                label=hier["label"],  edgecolor="white")
        ax.axvline(np.mean(flat[key]), color="blue",  linestyle="--", lw=2,
                   label=f"S1 mean={np.mean(flat[key]):.1f}")
        ax.axvline(np.mean(hier[key]), color="red",   linestyle="--", lw=2,
                   label=f"S2 mean={np.mean(hier[key]):.1f}")
        ax.set_title(title); ax.set_xlabel(xlabel); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results_comparison.png", dpi=140, bbox_inches="tight")
    print("\nSaved: results_comparison.png")

    flat_steps = np.mean(flat["steps"])
    hier_steps = np.mean(hier["steps"])
    reduction = (flat_steps - hier_steps) / flat_steps * 100
    print(f"Step reduction: {flat_steps:.1f} → {hier_steps:.1f}  ({reduction:+.1f}%)")


if __name__ == "__main__":
    flat = evaluate_flat(n_episodes=50)
    hier = evaluate_hierarchical(n_episodes=50)
    plot_comparison(flat, hier)
