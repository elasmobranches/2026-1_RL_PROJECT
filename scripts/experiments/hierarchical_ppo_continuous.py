"""연속 PPO 하위 정책과 Greedy/DQN 상위 선택기를 비교하는 계층형 실험."""
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env.hierarchical_continuous.continuous_lane_executor_env import ContinuousLaneExecutorEnv
from env.hierarchical_continuous.high_level_continuous_env import HighLevelContinuousEnv


class RandomLaneWrapper(gym.Wrapper):
    """매 에피소드 시작 시 임의의 목표 레인을 지정한다."""
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
        return self.env.reset(**kwargs)


def make_ll_env():
    env = ContinuousLaneExecutorEnv(n_beds=3, field_height=5)
    env.curriculum_level = 2
    env = RandomLaneWrapper(env)
    return Monitor(env)


def train_ppo_ll(total_timesteps=1_000_000, save_path="models/hppo_cont_ll"):
    os.makedirs("models", exist_ok=True)

    # PPO는 16개 환경에서 rollout을 병렬 수집한다.
    # 관측이 이미 [-1, 1]이므로 상위 환경과의 정규화 불일치를 피한다.
    vec_env = DummyVecEnv([make_ll_env for _ in range(16)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        gae_lambda=0.95,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=0,
    )

    print("=== PPO Low-level (continuous lane executor, n_envs=16) ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.close()
    print(f"PPO LL saved: {save_path}.zip")
    return model


def greedy_select(hl_env) -> int:
    ax = hl_env.inner.robot_pos[0]
    unfinished = [lx for lx in hl_env.lane_cols
                  if not hl_env._is_lane_already_done(lx)]
    if not unfinished:
        return 0
    nearest = min(unfinished, key=lambda lx: abs(lx - ax))
    return hl_env.lane_cols.index(nearest)


def evaluate(model, label, n_episodes=20):
    import numpy as np
    env = HighLevelContinuousEnv(model, n_beds=3, field_height=5)
    succs, covs, steps_l, visits_l = [], [], [], []
    for seed in range(n_episodes):
        obs, _ = env.reset(seed=seed)
        done, total_steps, visits = False, 0, 0
        while not done:
            action = greedy_select(env)
            obs, r, t, tr, info = env.step(action)
            total_steps += info["steps_for_lane"]
            visits += 1
            done = t or tr
        succs.append(info["coverage"] >= 1.0)
        covs.append(info["coverage"])
        steps_l.append(total_steps)
        visits_l.append(visits)

    print(f"\n[{label}]")
    print(f"  Success:  {np.mean(succs):.1%}")
    print(f"  Coverage: {np.mean(covs):.1%}")
    print(f"  Steps:    {np.mean(steps_l):.0f} ± {np.std(steps_l):.0f}")
    print(f"  Visits:   {np.mean(visits_l):.1f}")
    return np.mean(succs), np.mean(covs), np.mean(steps_l)


if __name__ == "__main__":
    model = train_ppo_ll(total_timesteps=1_000_000)
    evaluate(model, "PPO LL + Greedy HL")
