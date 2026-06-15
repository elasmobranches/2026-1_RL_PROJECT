"""Step 3: 목표 도달 보상을 쓰는 하위 정책과 거리 관측 DQN 상위 정책을 학습한다."""
import os
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.hierarchical.lane_executor_env import LaneExecutorEnv
from env.hierarchical.high_level_env import HighLevelFarmEnv


def mask_fn(env):
    return env.action_masks()


class RandomLaneWrapper(gym.Wrapper):
    """매 에피소드마다 하위 정책이 처리할 목표 레인을 무작위로 바꾼다."""
    def __init__(self, env):
        super().__init__(env)
        self._rng = np.random.default_rng()

    def reset(self, **kwargs):
        if kwargs.get("seed") is not None:
            self._rng = np.random.default_rng(kwargs["seed"])
        lane_idx = int(self._rng.integers(len(self.unwrapped.lane_cols)))
        target_col = self.unwrapped.lane_cols[lane_idx]
        if not kwargs.get("options"):
            kwargs["options"] = {}
        kwargs["options"]["target_lane_col"] = target_col
        return self.env.reset(**kwargs)

    def action_masks(self):
        return self.env.action_masks()


# 1단계: 목표 레인 도달 보상을 포함한 하위 정책 학습

def make_lane_env(seed=0):
    def _init():
        env = LaneExecutorEnv(n_beds=4, field_height=8)
        env = RandomLaneWrapper(env)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init


def train_low_level(total_timesteps=700_000, save_path="models/lane_executor_s3"):
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
    print("=== Step 3 Phase 1: Low-level with Goal-reaching reward ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.close()
    print(f"Low-level model saved to {save_path}.zip")
    return model


# 2단계: 거리 관측을 사용하는 이산 Off-policy DQN 상위 정책 학습

def make_hl_env(low_level_model, seed=0):
    def _init():
        env = HighLevelFarmEnv(
            low_level_model, n_beds=4, field_height=8, include_distances=True
        )
        env = Monitor(env)
        return env
    return _init


def train_high_level_dqn(low_level_model, total_timesteps=30_000, save_path="models/high_level_s3"):
    # 상위 step마다 하위 rollout 전체를 실행하며 DQN은 단일 환경을 사용한다.
    vec_env = DummyVecEnv([make_hl_env(low_level_model, seed=0)])

    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-3,
        buffer_size=10_000,
        learning_starts=500,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
        seed=0,
    )
    print("=== Step 3 Phase 2: High-level DQN (off-policy lane sequencing) ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.close()
    print(f"High-level DQN model saved to {save_path}.zip")
    return model


def train_high_level_ppo(low_level_model, total_timesteps=50_000, save_path="models/high_level_s3_ppo"):
    """작은 이산 문제에서 DQN과 비교하기 위한 PPO 상위 정책 대안."""
    from stable_baselines3 import PPO
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
    print("=== Step 3 Phase 2b: High-level PPO ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.close()
    print(f"High-level PPO model saved to {save_path}.zip")
    return model


if __name__ == "__main__":
    ll_model = train_low_level(total_timesteps=700_000)
    hl_model = train_high_level_dqn(ll_model, total_timesteps=300_000)
    print("=== Step 3 training complete ===")
