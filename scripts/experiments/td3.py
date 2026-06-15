"""연속 환경에서 TD3와 SAC의 탐색 성능을 비교하는 실험.

TD3는 엔트로피 항 없이 결정론적 정책, twin critic, 지연 정책 업데이트를
사용하므로 탐색이 중요한 본 문제에서 SAC와 차이가 있는지 확인한다.
"""
import os
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from env.continuous_farm_env_curriculum import ContinuousFarmEnvCurriculum


def make_env():
    env = ContinuousFarmEnvCurriculum(n_beds=3, field_height=5)
    env.curriculum_level = 2
    return Monitor(env)


def train(total_timesteps=1_500_000, save_path="models/td3_continuous"):
    os.makedirs("models", exist_ok=True)
    vec_env = DummyVecEnv([make_env for _ in range(4)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=5.0)

    # TD3에는 SAC의 엔트로피 항이 없으므로 탐색 노이즈를 명시적으로 추가한다.
    n_actions = vec_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.2 * np.ones(n_actions),  # 중간 수준의 탐색 노이즈
    )

    model = TD3(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=2_000,
        batch_size=4096,
        tau=0.005,
        gamma=0.99,
        train_freq=16,         # 16스텝마다 업데이트하여 처리량 향상
        gradient_steps=1,
        policy_delay=2,        # TD3의 지연 정책 업데이트
        target_noise_clip=0.5,
        action_noise=action_noise,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=0,
    )

    print("=== TD3 on ContinuousFarmEnvCurriculum ===")
    print("  Action noise σ=0.2, policy_delay=2, train_freq=16, bs=4096")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.save("models/td3_vecnorm.pkl")
    vec_env.close()
    print(f"Saved: {save_path}.zip")
    return model


if __name__ == "__main__":
    train()
