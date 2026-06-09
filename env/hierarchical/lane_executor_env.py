from __future__ import annotations
import numpy as np
from gymnasium import spaces
from env.farm_env import FarmEnv
from env.constants import (
    CELL_CROP, DONE_STATES,
    MOVE_DELTA,
    ACT_SCOUT, ACT_HARVEST, ACT_PEST,
    REWARD_STEP, REWARD_COLLISION,
    REWARD_SCOUT_NEW, REWARD_NORMAL_CONFIRM,
    REWARD_HARVEST, REWARD_PEST,
    REWARD_LANE_COMPLETE,
)


class LaneExecutorEnv(FarmEnv):
    """
    Low-level env: navigate to target lane and process all adjacent crops.

    Extends FarmEnv with:
    - ch4 (target lane indicator) added to observation → 5*H*W obs
    - Terminates when all crops adjacent to target_lane_col are in DONE_STATES
    - max_steps_per_lane replaces max_steps for episode limit
    """

    def __init__(
        self,
        n_beds: int = 4,
        field_height: int = 8,
        max_steps_per_lane: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__(n_beds=n_beds, field_height=field_height, render_mode=render_mode)

        self.lane_cols: list[int] = [c for c in range(1, self.W - 1) if (c - 1) % 3 == 0]
        self.max_steps_per_lane: int = max_steps_per_lane or self.H * self.W
        self.target_lane_col: int = self.lane_cols[0]

        # Override observation space: 5 channels
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5 * self.H * self.W,), dtype=np.float32
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed)
        self.target_lane_col = (options or {}).get("target_lane_col", self.lane_cols[0])
        self.step_count = 0
        return self._get_obs(), info

    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()          # 4 * H * W
        ch4 = np.zeros((self.H, self.W), dtype=np.float32)
        ch4[:, self.target_lane_col] = 1.0
        return np.concatenate([base, ch4.ravel()])

    def step(self, action: int):
        self.step_count += 1
        reward = REWARD_STEP

        if action in MOVE_DELTA:
            reward += self._handle_move(action)
        elif action == ACT_SCOUT:
            reward += self._handle_scout()
        elif action == ACT_HARVEST:
            reward += self._handle_harvest()
        elif action == ACT_PEST:
            reward += self._handle_pest()

        lane_done = self._is_lane_complete()
        if lane_done:
            reward += REWARD_LANE_COMPLETE

        terminated = lane_done
        truncated = (self.step_count >= self.max_steps_per_lane) and not terminated

        if self.render_mode == "human":
            self.render()

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            {
                "coverage": self._coverage_rate(),
                "lane_coverage": self._lane_coverage_rate(),
                "step": self.step_count,
            },
        )

    def _is_lane_complete(self) -> bool:
        return all(
            self.crop_states[r, c] in DONE_STATES
            for (r, c) in self._adjacent_lane_crops(self.target_lane_col)
        )

    def _adjacent_lane_crops(self, lane_col: int) -> list[tuple[int, int]]:
        result = []
        for row in range(2, self.H - 2):
            for dc in (-1, 1):
                nc = lane_col + dc
                if 0 <= nc < self.W and self.layout[row, nc] == CELL_CROP:
                    result.append((row, nc))
        return result

    def _lane_coverage_rate(self) -> float:
        crops = self._adjacent_lane_crops(self.target_lane_col)
        if not crops:
            return 1.0
        done = sum(1 for (r, c) in crops if self.crop_states[r, c] in DONE_STATES)
        return done / len(crops)
