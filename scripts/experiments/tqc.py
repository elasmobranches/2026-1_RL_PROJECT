"""연속 환경에서 분포형 가치 추정을 사용하는 TQC 비교 실험.

TQC는 SAC에 quantile regression을 결합해 Q값 과대추정을 줄이는 알고리즘으로,
가치 추정 안정성이 최종 성능에 미치는 영향을 확인한다.
"""
import os
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.continuous_farm_env_curriculum import ContinuousFarmEnvCurriculum


def make_env():
    env = ContinuousFarmEnvCurriculum(n_beds=3, field_height=5)
    env.curriculum_level = 2
    return Monitor(env)


def train(total_timesteps=1_500_000, save_path="models/tqc_continuous"):
    os.makedirs("models", exist_ok=True)
    vec_env = DummyVecEnv([make_env for _ in range(4)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=5.0)

    model = TQC(
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
        top_quantiles_to_drop_per_net=2,  # 상위 quantile을 제거해 과대추정 완화
        policy_kwargs={"net_arch": [256, 256], "n_critics": 2},
        verbose=1,
        seed=0,
    )

    print("=== TQC on ContinuousFarmEnvCurriculum ===")
    print("  quantiles=25, drop_top=2, train_freq=16, bs=4096")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.save("models/tqc_vecnorm.pkl")
    vec_env.close()
    print(f"Saved: {save_path}.zip")
    return model


if __name__ == "__main__":
    train()
