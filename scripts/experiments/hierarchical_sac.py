"""연속 SAC 하위 정책과 DQN 상위 정책을 결합한 계층형 비교 실험."""
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.hierarchical_continuous.continuous_lane_executor_env import ContinuousLaneExecutorEnv
from env.hierarchical_continuous.high_level_continuous_env import HighLevelContinuousEnv


def mask_fn(env):
    return None  # 연속 행동을 사용하는 SAC에는 마스킹이 필요하지 않다.


# 1단계: 연속 SAC 하위 정책 학습

class RandomLaneWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._rng = np.random.default_rng()

    def reset(self, **kwargs):
        if kwargs.get("seed") is not None:
            self._rng = np.random.default_rng(kwargs["seed"])
        lane_idx = int(self._rng.integers(len(self.unwrapped.lane_x)))
        self.unwrapped.target_lane_x = self.unwrapped.lane_x[lane_idx]
        self.unwrapped._goal_reached_cont = False
        if not kwargs.get("options"):
            kwargs["options"] = {}
        kwargs["options"]["target_lane_col"] = 2  # 레벨 2 헤드랜드 시작
        return self.env.reset(**kwargs)

    def action_masks(self):
        return None


def make_ll_env():
    env = ContinuousLaneExecutorEnv(n_beds=3, field_height=5)
    env.curriculum_level = 2
    env = RandomLaneWrapper(env)
    return Monitor(env)


def train_sac_ll(total_timesteps=1_000_000, save_path="models/hsac_ll"):
    os.makedirs("models", exist_ok=True)
    # 관측은 이미 [-1, 1]이므로 VecNormalize를 사용하지 않는다.
    # 상위 환경이 정규화 불일치 없이 내부 관측을 직접 사용할 수 있다.
    vec_env = DummyVecEnv([make_ll_env for _ in range(4)])

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        buffer_size=400_000,
        learning_starts=2_000,
        batch_size=4096,
        tau=0.005,
        gamma=0.99,
        train_freq=16,        # 업데이트마다 16스텝을 수집해 처리량 향상
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=0,
    )

    print("=== Hierarchical SAC Phase 1: SAC Low-level (continuous lane executor) ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.close()
    print(f"SAC LL saved: {save_path}.zip")
    return model


# 2단계: 다음 레인을 선택하는 DQN 상위 정책 학습

def make_hl_env(ll_model):
    def _init():
        env = HighLevelContinuousEnv(ll_model, n_beds=3, field_height=5)
        return Monitor(env)
    return _init


def train_dqn_hl(ll_model, total_timesteps=30_000, save_path="models/hsac_hl"):
    vec_env = DummyVecEnv([make_hl_env(ll_model)])

    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-3,
        buffer_size=5_000,
        learning_starts=300,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=300,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
        seed=0,
    )

    print("=== Hierarchical SAC Phase 2: DQN High-level (lane sequencing) ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.close()
    print(f"DQN HL saved: {save_path}.zip")
    return model


if __name__ == "__main__":
    ll = train_sac_ll(total_timesteps=1_000_000)
    train_dqn_hl(ll, total_timesteps=30_000)
    print("=== Hierarchical SAC training complete ===")
