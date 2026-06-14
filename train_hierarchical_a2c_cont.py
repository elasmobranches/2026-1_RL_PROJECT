# train_hierarchical_a2c_cont.py
# Hierarchical A2C (continuous): A2C low-level + Greedy high-level
#
# A2C vs PPO:
#   - No clipping → more aggressive updates
#   - n_epochs=1 always (single update per rollout)
#   - Uses RMSProp by default (original A3C optimizer)
#   - Typically needs smaller learning rate and shorter n_steps
#
# Why A2C on continuous lane executor:
#   - Direct comparison against PPO LL and SAC LL
#   - On-policy like PPO, so n_envs=16 is valid
#   - No VecNormalize (obs already in [-1,1]; avoids HL obs mismatch)
import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env.hierarchical_continuous.continuous_lane_executor_env import ContinuousLaneExecutorEnv
from env.hierarchical_continuous.high_level_continuous_env import HighLevelContinuousEnv


class RandomLaneWrapper(gym.Wrapper):
    """Assigns a random target lane at each episode reset."""
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


def benchmark_fps(n_steps_val: int, n_envs: int) -> float:
    """Measure actual fps before committing to full training."""
    vec_env = DummyVecEnv([make_ll_env for _ in range(n_envs)])
    model = A2C(
        "MlpPolicy", vec_env,
        n_steps=n_steps_val, learning_rate=7e-4,
        verbose=0, seed=0,
    )
    model.learn(total_timesteps=n_steps_val * n_envs * 3)  # warmup 3 updates
    t0 = time.time()
    model.learn(total_timesteps=n_steps_val * n_envs * 10, reset_num_timesteps=False)
    fps = (n_steps_val * n_envs * 10) / (time.time() - t0)
    vec_env.close()
    return fps


def train(total_timesteps: int = 1_000_000,
          save_path: str = "models/ha2c_cont_ll") -> A2C:
    os.makedirs("models", exist_ok=True)

    # A2C hyperparameters:
    # n_steps=256: balanced between stability (longer) and frequency (shorter)
    # learning_rate=7e-4: standard A2C/A3C rate; lower than PPO (3e-4) for stability
    # gae_lambda=1.0: use full Monte Carlo returns (original A3C style, no GAE)
    # ent_coef=0.01: encourage exploration
    # vf_coef=0.25: value function loss weight (lower than PPO's 0.5)
    # normalize_advantage=False: keep raw advantages (A2C original)
    # use_rms_prop=True: RMSProp as in original A3C paper
    # rms_prop_eps=1e-5: numerical stability for RMSProp
    N_ENVS = 16
    N_STEPS = 256

    vec_env = DummyVecEnv([make_ll_env for _ in range(N_ENVS)])

    model = A2C(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=N_STEPS,
        learning_rate=7e-4,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.25,
        max_grad_norm=0.5,
        use_rms_prop=True,
        rms_prop_eps=1e-5,
        normalize_advantage=False,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=0,
    )

    print(f"=== A2C Low-level (continuous lane executor) ===")
    print(f"  n_envs={N_ENVS}, n_steps={N_STEPS}, lr=7e-4, RMSProp")
    print(f"  total_timesteps={total_timesteps:,}")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.close()
    print(f"A2C LL saved: {save_path}.zip")
    return model


def greedy_select(hl_env: HighLevelContinuousEnv) -> int:
    ax = hl_env.inner.robot_pos[0]
    unfinished = [lx for lx in hl_env.lane_cols
                  if not hl_env._is_lane_already_done(lx)]
    if not unfinished:
        return 0
    nearest = min(unfinished, key=lambda lx: abs(lx - ax))
    return hl_env.lane_cols.index(nearest)


def evaluate(model: A2C,
             label: str = "A2C LL + Greedy HL",
             n_episodes: int = 20) -> dict:
    env = HighLevelContinuousEnv(model, n_beds=3, field_height=5)
    succs, covs, steps_l, visits_l = [], [], [], []

    for seed in range(n_episodes):
        obs, _ = env.reset(seed=seed)
        done, total_steps, visits = False, 0, 0
        while not done:
            action = greedy_select(env)
            obs, _, t, tr, info = env.step(action)
            total_steps += info["steps_for_lane"]
            visits += 1
            done = t or tr

        succs.append(info["coverage"] >= 1.0)
        covs.append(info["coverage"])
        steps_l.append(total_steps)
        visits_l.append(visits)

    result = {
        "success": float(np.mean(succs)),
        "coverage": float(np.mean(covs)),
        "steps_mean": float(np.mean(steps_l)),
        "steps_std": float(np.std(steps_l)),
        "visits": float(np.mean(visits_l)),
    }
    print(f"\n[{label}]")
    print(f"  Success:  {result['success']:.1%}")
    print(f"  Coverage: {result['coverage']:.1%}")
    print(f"  Steps:    {result['steps_mean']:.0f} ± {result['steps_std']:.0f}")
    print(f"  Visits:   {result['visits']:.1f}")
    return result


if __name__ == "__main__":
    # Benchmark fps before full training
    print("Benchmarking A2C fps...")
    fps = benchmark_fps(n_steps_val=256, n_envs=16)
    eta_min = 1_000_000 / fps / 60
    print(f"A2C fps: {fps:.0f}  →  1M steps ≈ {eta_min:.0f}분\n")

    model = train(total_timesteps=1_000_000)
    evaluate(model)
