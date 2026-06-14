"""
HighLevelContinuousEnv: DQN high-level for Hierarchical SAC.

Same interface as HighLevelFarmEnv (Step 3) but wraps the continuous
SAC low-level instead of the discrete MaskablePPO low-level.

obs = [lane_done_rates(n_lanes), normalized_distances(n_lanes)]  → 10-dim
action = lane index (Discrete n_lanes)
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.continuous_farm_env import CELL_SIZE, DONE_STATES
from env.hierarchical_continuous.continuous_lane_executor_env import ContinuousLaneExecutorEnv


class HighLevelContinuousEnv(gym.Env):

    def __init__(self, sac_ll_model, n_beds=3, field_height=5, max_lane_visits=None):
        super().__init__()
        self.ll = sac_ll_model
        self.inner = ContinuousLaneExecutorEnv(n_beds=n_beds, field_height=field_height)
        self.inner.curriculum_level = 2

        self.lane_cols = self.inner.lane_x   # continuous x positions
        self.n_lanes = len(self.lane_cols)
        self.max_lane_visits = max_lane_visits or self.n_lanes * 3
        self._lane_visits = 0

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_lanes * 2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_lanes)

    def reset(self, seed=None, options=None):
        self.inner.reset(seed=seed)
        self._lane_visits = 0
        return self._get_hl_obs(), {}

    def _get_hl_obs(self) -> np.ndarray:
        completion = np.zeros(self.n_lanes, dtype=np.float32)
        distances  = np.zeros(self.n_lanes, dtype=np.float32)
        ax = float(self.inner.robot_pos[0])
        max_dist = float(self.inner.W_m)

        for i, lx in enumerate(self.lane_cols):
            crops = [j for j, (cx, _) in enumerate(self.inner.crop_centres)
                     if abs(cx - lx) < 1.5 * CELL_SIZE]
            if crops:
                done = sum(1 for j in crops if self.inner.crop_states[j] in DONE_STATES)
                completion[i] = done / len(crops)
            distances[i] = abs(ax - lx) / max_dist

        return np.concatenate([completion, distances])

    def _is_lane_already_done(self, lane_x: float) -> bool:
        crops = [j for j, (cx, _) in enumerate(self.inner.crop_centres)
                 if abs(cx - lane_x) < 1.5 * CELL_SIZE]
        return bool(crops) and all(self.inner.crop_states[j] in DONE_STATES for j in crops)

    def step(self, action: int):
        target_lx = self.lane_cols[action]
        was_done = self._is_lane_already_done(target_lx)

        self.inner.target_lane_x = target_lx
        self.inner.step_count = 0
        self.inner._goal_reached_cont = False
        self.inner._prev_potential = self.inner._potential()  # reset shaping baseline

        ll_obs = self.inner._get_obs()
        lane_done = lane_trunc = False
        total_steps = 0
        last_info: dict = {}

        while not (lane_done or lane_trunc):
            # SAC predict (no action masking needed — continuous actions)
            ll_action, _ = self.ll.predict(ll_obs, deterministic=True)
            ll_obs, _, lane_done, lane_trunc, last_info = self.inner.step(ll_action)
            total_steps += 1

        self._lane_visits += 1

        hl_reward = -total_steps * 0.01
        if lane_done and not was_done:
            hl_reward += 5.0

        all_done = self.inner._is_complete()
        if all_done:
            hl_reward += 20.0

        terminated = all_done
        truncated = (self._lane_visits >= self.max_lane_visits) and not terminated

        return self._get_hl_obs(), float(hl_reward), terminated, truncated, {
            "steps_for_lane": total_steps,
            "coverage": last_info.get("coverage", self.inner._coverage_rate()),
            "lane_visits": self._lane_visits,
        }
