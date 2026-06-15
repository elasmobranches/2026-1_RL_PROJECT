"""동일한 시드에서 Step 1~3 정책의 성공률·커버리지·스텝을 비교한다."""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO, DQN
from env.farm_env import FarmEnv
from env.hierarchical.high_level_env import HighLevelFarmEnv


def run_flat(model, env, seed):
    obs, _ = env.reset(seed=seed)
    r, steps, t, tr = 0.0, 0, False, False
    while not (t or tr):
        a, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, rew, t, tr, info = env.step(int(a))
        r += rew; steps += 1
    return r, steps, info["coverage"], t


def run_hl(hl_model, ll_model, seed, include_distances):
    env = HighLevelFarmEnv(
        ll_model,
        n_beds=4,
        field_height=8,
        include_distances=include_distances,
    )
    obs, _ = env.reset(seed=seed)
    r, total_steps, visits, t, tr = 0.0, 0, 0, False, False
    while not (t or tr):
        a, _ = hl_model.predict(obs, deterministic=True)
        obs, rew, t, tr, info = env.step(int(a))
        r += rew; total_steps += info["steps_for_lane"]; visits += 1
    return r, total_steps, info["coverage"], t, visits


def evaluate_all(n=50):
    results = {}

    # Step 1: 단일 MaskablePPO 정책
    flat_env = FarmEnv(n_beds=4, field_height=8)
    flat_m = MaskablePPO.load("models/farm_ppo")
    data = [run_flat(flat_m, flat_env, ep) for ep in range(n)]
    r, s, c, su = zip(*data)
    results["Step1 Flat PPO"] = dict(rewards=list(r), steps=list(s), covs=list(c), succs=list(su))
    print(f"\n[Step 1 - Flat PPO]")
    print(f"  Success: {np.mean(su):.1%}  Steps: {np.mean(s):.1f}±{np.std(s):.1f}  Coverage: {np.mean(c):.1%}")

    # Step 2: 완료율만 관측하는 PPO 상위 정책
    ll2 = MaskablePPO.load("models/lane_executor")
    hl2 = PPO.load("models/high_level")
    data = [run_hl(hl2, ll2, ep, include_distances=False) for ep in range(n)]
    r, s, c, su, v = zip(*data)
    results["Step2 Hierarchical PPO"] = dict(rewards=list(r), steps=list(s), covs=list(c), succs=list(su))
    print(f"\n[Step 2 - Hierarchical PPO]")
    print(f"  Success: {np.mean(su):.1%}  Steps: {np.mean(s):.1f}±{np.std(s):.1f}  Coverage: {np.mean(c):.1%}  Visits: {np.mean(v):.1f}")

    # Step 3: 완료율과 거리를 관측하는 DQN 상위 정책
    ll3 = MaskablePPO.load("models/lane_executor_s3")
    hl3 = DQN.load("models/high_level_s3")
    data = [run_hl(hl3, ll3, ep, include_distances=True) for ep in range(n)]
    r, s, c, su, v = zip(*data)
    results["Step3 Goal+DQN"] = dict(rewards=list(r), steps=list(s), covs=list(c), succs=list(su))
    print(f"\n[Step 3 - Goal-reaching + DQN]")
    print(f"  Success: {np.mean(su):.1%}  Steps: {np.mean(s):.1f}±{np.std(s):.1f}  Coverage: {np.mean(c):.1%}  Visits: {np.mean(v):.1f}")

    _plot(results)
    return results


def _plot(results):
    labels = list(results.keys())
    colors = ["steelblue", "coral", "mediumseagreen"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Step 1 vs Step 2 vs Step 3 Comparison", fontsize=13, fontweight="bold")

    for ax, key, xlabel, title in [
        (axes[0], "steps",   "Steps",          "Completion Steps"),
        (axes[1], "covs",    "Coverage Rate",   "Field Coverage"),
        (axes[2], "succs",   "Success (0/1)",   "Success Rate"),
    ]:
        for label, color in zip(labels, colors):
            data = results[label][key]
            if key == "succs":
                ax.bar(label.split()[0], np.mean(data)*100, color=color, alpha=0.8)
                ax.set_ylabel("Success Rate (%)")
            else:
                ax.hist(data, bins=12, alpha=0.6, color=color, label=f"{label.split()[0]} μ={np.mean(data):.1f}", edgecolor="white")
                ax.axvline(np.mean(data), color=color, linestyle="--", lw=2)
        if key != "succs":
            ax.legend(fontsize=8)
        ax.set_title(title); ax.set_xlabel(xlabel)

    plt.tight_layout()
    out = "assets/figures/results_step3_comparison.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    evaluate_all(n=50)
