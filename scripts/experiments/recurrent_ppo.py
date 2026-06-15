"""부분 관측 대응을 위해 이산 FarmEnv에서 RecurrentPPO(LSTM)를 비교한다.

LSTM은 에피소드 안의 예찰 이력을 기억할 수 있지만 RecurrentPPO는 행동
마스킹을 지원하지 않으므로 충돌 페널티만으로 무효 행동을 학습해야 한다.
"""
import os
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env.farm_env import FarmEnv


def make_env(seed=0):
    def _init():
    # ActionMasker 없이 충돌 페널티(-2.0)로 무효 행동을 억제한다.
        env = FarmEnv(n_beds=4, field_height=8)
        return Monitor(env)
    return _init


def train(total_timesteps=500_000, save_path="models/recurrent_ppo"):
    os.makedirs("models", exist_ok=True)
    # On-policy RecurrentPPO는 여러 환경에서 rollout을 병렬 수집할 수 있다.
    vec_env = DummyVecEnv([make_env(seed=i) for i in range(16)])

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        n_steps=512,           # 업데이트 전 환경별 수집 스텝
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        gae_lambda=0.95,
        policy_kwargs={
            "lstm_hidden_size": 128,
            "n_lstm_layers": 1,
            "net_arch": [128, 128],
        },
        verbose=1,
        seed=0,
    )

    print("=== RecurrentPPO (LSTM) on discrete FarmEnv ===")
    print("  16 parallel envs, LSTM hidden=128")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.close()
    print(f"Saved: {save_path}.zip")
    return model


if __name__ == "__main__":
    train()
