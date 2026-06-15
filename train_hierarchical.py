# train_hierarchical.py
import os
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.hierarchical.lane_executor_env import LaneExecutorEnv
from env.hierarchical.high_level_env import HighLevelFarmEnv


def mask_fn(env):
    return env.unwrapped.action_masks()


# ── Phase 1: Low-level (LaneExecutorEnv) ──────────────────────────────

class RandomLaneWrapper(gym.Wrapper):
    """Sets a random target lane at each episode reset."""
    def __init__(self, env):
        super().__init__(env)
        self._rng = np.random.default_rng()

    def reset(self, **kwargs):
        if kwargs.get("seed") is not None:
            self._rng = np.random.default_rng(kwargs["seed"])
        lane_idx = int(self._rng.integers(len(self.unwrapped.lane_cols)))
        target_col = self.unwrapped.lane_cols[lane_idx]
        options = kwargs.get("options") or {}
        options["target_lane_col"] = target_col
        kwargs["options"] = options
        return self.env.reset(**kwargs)


def make_lane_env(seed=0):
    def _init():
        env = LaneExecutorEnv(n_beds=4, field_height=8)
        env = RandomLaneWrapper(env)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init


def train_low_level(total_timesteps=500_000, save_path="models/lane_executor"):
    os.makedirs("models", exist_ok=True)
    vec_env = DummyVecEnv([make_lane_env(seed=i) for i in range(16)])

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        seed=0,
    )
    print("=== Phase 1: Training Low-level LaneExecutorEnv ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.close()
    print(f"Low-level model saved to {save_path}.zip")
    return model


# ── Phase 2: High-level (HighLevelFarmEnv) ────────────────────────────

def make_hl_env(low_level_model, seed=0):
    def _init():
        # Step 2 baseline: completion-only high-level observation.
        env = HighLevelFarmEnv(
            low_level_model, n_beds=4, field_height=8, include_distances=False
        )
        env = Monitor(env)
        return env
    return _init


def train_high_level(low_level_model, total_timesteps=50_000, save_path="models/high_level"):
    # HL env is slow (each step runs full LL rollout), keep n_envs small
    vec_env = DummyVecEnv([make_hl_env(low_level_model, seed=i) for i in range(4)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        seed=0,
    )
    print("=== Phase 2: Training High-level HighLevelFarmEnv ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.close()
    print(f"High-level model saved to {save_path}.zip")
    return model


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ll_model = train_low_level(total_timesteps=500_000)
    hl_model = train_high_level(ll_model, total_timesteps=200_000)
    print("=== Hierarchical training complete ===")
