"""Step 1: 단일 MaskablePPO 정책을 이산 FarmEnv에서 학습한다."""
import os
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.farm_env import FarmEnv


def mask_fn(env):
    return env.action_masks()


def make_env(seed=0):
    def _init():
        env = FarmEnv(n_beds=4, field_height=8)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    vec_env = DummyVecEnv([make_env(seed=i) for i in range(4)])

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        seed=42,
    )

    print("Training started...")
    model.learn(total_timesteps=500_000)

    model.save("models/farm_ppo")
    vec_env.close()
    print("Model saved to models/farm_ppo.zip")


if __name__ == "__main__":
    main()
