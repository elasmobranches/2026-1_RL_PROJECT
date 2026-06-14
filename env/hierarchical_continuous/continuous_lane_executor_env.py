"""
ContinuousLaneExecutorEnv: SAC low-level for Hierarchical SAC.

Extended from ContinuousFarmEnvCurriculum with:
- target_lane_x: x-coordinate of the lane to process
- obs += target lane relative position (1 extra dim → 29-dim total)
- terminates when all crops adjacent to target_lane_x are done
- goal-reaching bonus on first arrival at target lane
"""
from __future__ import annotations
import numpy as np
from gymnasium import spaces
from env.continuous_farm_env_curriculum import ContinuousFarmEnvCurriculum
from env.continuous_farm_env import CELL_SIZE, DONE_STATES, MAX_STEPS

REWARD_GOAL_REACH_CONT = 2.0   # bonus: first arrival at target lane x


class ContinuousLaneExecutorEnv(ContinuousFarmEnvCurriculum):

    def __init__(self, n_beds=3, field_height=5, max_steps=MAX_STEPS, render_mode=None):
        super().__init__(n_beds=n_beds, field_height=field_height,
                         max_steps=max_steps, render_mode=render_mode)
        self.target_lane_x: float = self.lane_x[0]
        self._goal_reached_cont: bool = False

        # obs: parent 28-dim + target_lane_dx (1) = 29-dim
        parent_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(parent_dim + 1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._goal_reached_cont = False
        return self._get_obs(), info

    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()                              # 28-dim float32
        dx = np.float32((self.target_lane_x - self.robot_pos[0]) / self.W_m)
        return np.clip(np.append(base, dx).astype(np.float32), -1.0, 1.0)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Goal-reaching: first time robot arrives at target lane column
        if not self._goal_reached_cont:
            dist_to_lane = abs(self.robot_pos[0] - self.target_lane_x)
            if dist_to_lane < 0.8 * CELL_SIZE:
                reward += REWARD_GOAL_REACH_CONT
                self._goal_reached_cont = True

        # Override termination: done when target lane is fully processed
        lane_done = self._is_target_lane_done()
        if lane_done:
            reward += 10.0   # lane completion bonus (same magnitude as discrete LaneExecutorEnv)
        terminated = lane_done
        truncated = (self.step_count >= self.max_steps) and not lane_done

        info['lane_coverage'] = self._target_lane_coverage()
        return self._get_obs(), float(reward), terminated, truncated, info

    def _target_lane_crops(self) -> list[int]:
        """Indices of crops within ±1.5 cells of target lane x."""
        return [
            i for i, (cx, _) in enumerate(self.crop_centres)
            if abs(cx - self.target_lane_x) < 1.5 * CELL_SIZE
        ]

    def _is_target_lane_done(self) -> bool:
        return all(self.crop_states[i] in DONE_STATES
                   for i in self._target_lane_crops())

    def _target_lane_coverage(self) -> float:
        crops = self._target_lane_crops()
        if not crops:
            return 1.0
        return sum(1 for i in crops if self.crop_states[i] in DONE_STATES) / len(crops)
