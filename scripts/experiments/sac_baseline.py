# train_sac.py — Step 4: SAC on ContinuousFarmEnv
import os
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.continuous_farm_env import ContinuousFarmEnv


def make_env(seed=0):
    def _init():
        env = ContinuousFarmEnv(n_beds=3, field_height=5)
        return Monitor(env)
    return _init


def train(total_timesteps=500_000, save_path="models/sac_continuous"):
    os.makedirs("models", exist_ok=True)
    vec_env = DummyVecEnv([make_env(seed=i) for i in range(1)])
    # norm_obs=True helps with 124-dim obs; norm_reward=False to avoid distorting sparse rewards
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=5.0)

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-3,          # faster convergence
        buffer_size=200_000,
        learning_starts=2_000,       # start learning sooner
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=2,            # more updates per step
        ent_coef="auto",
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=0,
    )

    print("=== Step 4: SAC on ContinuousFarmEnv (proximity shaping) ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.save("models/sac_vecnormalize.pkl")
    vec_env.close()
    print(f"SAC model saved to {save_path}.zip")
    return model


if __name__ == "__main__":
    train()
