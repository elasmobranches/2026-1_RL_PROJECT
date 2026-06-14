# train_recurrent_ppo.py — RecurrentPPO (LSTM) on discrete FarmEnv
# Motivation: partial observability (hidden crop states until scouted)
# → LSTM can remember scouting history within an episode
# Note: RecurrentPPO does NOT support action masking → use soft collision penalty
import os
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env.farm_env import FarmEnv


def make_env(seed=0):
    def _init():
        # No ActionMasker — RecurrentPPO handles continuous/discrete natively
        # Collision penalty (-2.0) discourages invalid actions organically
        env = FarmEnv(n_beds=4, field_height=8)
        return Monitor(env)
    return _init


def train(total_timesteps=500_000, save_path="models/recurrent_ppo"):
    os.makedirs("models", exist_ok=True)
    # RecurrentPPO supports n_envs > 1 (on-policy → fast parallel collection)
    vec_env = DummyVecEnv([make_env(seed=i) for i in range(16)])

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        n_steps=512,           # steps per env before update
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
