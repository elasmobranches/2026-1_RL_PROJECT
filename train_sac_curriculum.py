# train_sac_curriculum.py — SAC + Curriculum Learning
import os
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.continuous_farm_env_curriculum import ContinuousFarmEnvCurriculum, CurriculumCallback


def make_env():
    env = ContinuousFarmEnvCurriculum(n_beds=3, field_height=5)
    env.curriculum_level = 2   # always headland start — skip curriculum phases
    return Monitor(env)


def train(total_timesteps=2_000_000, save_path="models/sac_curriculum"):
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

    callback = CurriculumCallback(
        vec_env,
        success_threshold=0.4,   # 40% coverage to count as success
        window=15,
        level_up_at=0.6,         # 60% of recent episodes must succeed
        verbose=1,
    )

    print("=== SAC + Curriculum (Level 0→1→2) ===")
    print("  Level 0: spawn next to crop")
    print("  Level 1: spawn in field lane")
    print("  Level 2: spawn at headland (full episode)")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(save_path)
    vec_env.save("models/sac_curriculum_vecnorm.pkl")
    vec_env.close()
    print(f"Saved: {save_path}.zip")


if __name__ == "__main__":
    train()
