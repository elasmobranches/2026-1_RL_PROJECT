# train_sac_large.py — Step 4 확장: n_beds=4 (더 큰 연속 환경)
import os
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.continuous_farm_env_curriculum import ContinuousFarmEnvCurriculum


def make_env():
    env = ContinuousFarmEnvCurriculum(n_beds=4, field_height=8)
    env.curriculum_level = 2
    return Monitor(env)


def train(total_timesteps=1_500_000, save_path="models/sac_large"):
    os.makedirs("models", exist_ok=True)
    # n_envs=4: env 수집 병렬화 (SAC는 off-policy라 n_envs>1 가능)
    # train_freq=16: 4env × 4steps = 16 transitions 수집 후 1 gradient update
    # batch_size=4096: RTX 3080 GPU 최대 활용
    vec_env = DummyVecEnv([make_env for _ in range(4)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=5.0)
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=2_000,
        batch_size=4096,
        tau=0.005,
        gamma=0.99,
        train_freq=16,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=0,
    )
    print("=== SAC Large Map (n_beds=4, field_height=8) ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.save("models/sac_large_vecnorm.pkl")
    vec_env.close()
    print(f"Saved: {save_path}.zip")


if __name__ == "__main__":
    train()
