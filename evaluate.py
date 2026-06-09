# evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env.farm_env import FarmEnv
from env.constants import (
    CELL_PATH, CELL_CROP, CELL_WALL,
    STATE_UNKNOWN, STATE_NORMAL_DONE, STATE_HARVEST_PENDING,
    STATE_PEST_PENDING, STATE_HARVEST_DONE, STATE_PEST_DONE,
)


def mask_fn(env):
    return env.action_masks()


def run_episode(model, env_raw, seed=0, render=False):
    obs, _ = env_raw.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True, action_masks=env_raw.action_masks())
        obs, reward, terminated, truncated, info = env_raw.step(int(action))
        total_reward += reward
        steps += 1
        if render:
            env_raw.render()

    return total_reward, steps, info["coverage"], terminated


def evaluate(model_path="models/farm_ppo", n_episodes=50):
    env = FarmEnv(n_beds=4, field_height=8)
    model = MaskablePPO.load(model_path)

    rewards, steps_list, coverages, successes = [], [], [], []
    for ep in range(n_episodes):
        r, s, cov, term = run_episode(model, env, seed=ep)
        rewards.append(r)
        steps_list.append(s)
        coverages.append(cov)
        successes.append(term)

    print(f"\n=== Evaluation over {n_episodes} episodes ===")
    print(f"Success rate:      {np.mean(successes):.1%}")
    print(f"Avg reward:        {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Avg coverage:      {np.mean(coverages):.1%}")
    print(f"Avg steps:         {np.mean(steps_list):.1f}")

    _plot_results(rewards, steps_list, coverages, successes)


def _plot_results(rewards, steps_list, coverages, successes):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(rewards, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("Episode Reward Distribution")
    axes[0].set_xlabel("Total Reward")
    axes[0].set_ylabel("Count")

    axes[1].hist(coverages, bins=20, color="mediumseagreen", edgecolor="white")
    axes[1].set_title("Coverage Rate Distribution")
    axes[1].set_xlabel("Coverage (%)")
    axes[1].axvline(np.mean(coverages), color="red", linestyle="--", label=f"mean={np.mean(coverages):.1%}")
    axes[1].legend()

    axes[2].hist(steps_list, bins=20, color="coral", edgecolor="white")
    axes[2].set_title("Steps to Completion")
    axes[2].set_xlabel("Steps")

    plt.tight_layout()
    plt.savefig("results_evaluation.png", dpi=150)
    print("Saved: results_evaluation.png")
    plt.show()


def visualize_single_episode(model_path="models/farm_ppo", seed=0):
    """단일 에피소드를 matplotlib으로 시각화."""
    env = FarmEnv(n_beds=4, field_height=8)
    model = MaskablePPO.load(model_path)

    STATE_COLORS = {
        STATE_UNKNOWN:         [0.8, 0.8, 0.8],
        STATE_NORMAL_DONE:     [0.2, 0.8, 0.2],
        STATE_HARVEST_PENDING: [1.0, 0.9, 0.0],
        STATE_PEST_PENDING:    [1.0, 0.4, 0.1],
        STATE_HARVEST_DONE:    [0.0, 0.5, 0.0],
        STATE_PEST_DONE:       [0.6, 0.2, 0.0],
    }
    CELL_COLORS = {CELL_PATH: [0.95, 0.95, 0.95], CELL_WALL: [0.2, 0.2, 0.2]}

    obs, _ = env.reset(seed=seed)
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.ion()
    done = False

    while not done:
        img = np.ones((env.H, env.W, 3))
        for r in range(env.H):
            for c in range(env.W):
                if env.layout[r, c] == CELL_CROP:
                    img[r, c] = STATE_COLORS[env.crop_states[r, c]]
                else:
                    img[r, c] = CELL_COLORS[env.layout[r, c]]
        ra, ca = env.agent_pos
        img[ra, ca] = [0.1, 0.1, 1.0]  # agent = blue

        ax.clear()
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"Step {env.step_count} | Coverage {env._coverage_rate():.1%}")
        plt.pause(0.05)

        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    plt.ioff()
    plt.savefig("episode_final_state.png", dpi=150)
    print("Saved: episode_final_state.png")
    plt.show()


if __name__ == "__main__":
    evaluate()
    visualize_single_episode()
