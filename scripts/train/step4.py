"""Step 4: 단순화 관측과 고정 헤드랜드 시작 조건에서 SAC를 학습한다."""
import os
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.continuous_farm_env_curriculum import ContinuousFarmEnvCurriculum


MODEL_PATH = "models/sac_simplified"
VECNORM_PATH = "models/sac_simplified_vecnorm.pkl"


def make_env():
    env = ContinuousFarmEnvCurriculum(n_beds=3, field_height=5)
    env.curriculum_level = 2   # 커리큘럼 단계를 건너뛰고 항상 헤드랜드에서 시작
    return Monitor(env)


def train(total_timesteps=2_000_000, save_path=MODEL_PATH, vecnorm_path=VECNORM_PATH):
    os.makedirs("models", exist_ok=True)
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=5.0)

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=0,
    )

    print("=== SAC + Simplified Observation (fixed Level 2 start) ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.save(vecnorm_path)
    vec_env.close()
    print(f"Saved: {save_path}.zip")


if __name__ == "__main__":
    train()
