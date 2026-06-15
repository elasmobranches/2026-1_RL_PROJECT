from __future__ import annotations
import numpy as np
from gymnasium import spaces
from env.farm_env import FarmEnv
from env.constants import (
    CELL_CROP, DONE_STATES,
    MOVE_DELTA,
    ACT_SCOUT, ACT_HARVEST, ACT_PEST,
    REWARD_COLLISION,
    REWARD_SCOUT_NEW, REWARD_NORMAL_CONFIRM,
    REWARD_HARVEST, REWARD_PEST,
    REWARD_LANE_COMPLETE, REWARD_LANE_STEP,
    REWARD_GOAL_REACH,
)


class LaneExecutorEnv(FarmEnv):
    """
    Step 2·3의 하위 정책이 사용하는 목표 레인 실행 환경.

    FarmEnv에 목표 레인을 표시하는 ch4를 추가하여 5*H*W 관측을 만든다.
    목표 레인에 인접한 모든 작물이 완료 상태가 되면 에피소드를 종료하며,
    max_steps_per_lane을 레인 한 번 처리의 시간 제한으로 사용한다.
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

        # 부모 환경의 4채널 관측에 목표 레인 채널을 추가한다.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5 * self.H * self.W,), dtype=np.float32
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed)
        self.target_lane_col = (options or {}).get("target_lane_col", self.lane_cols[0])
        self.step_count = 0
        self._goal_reached: bool = False   # 목표 레인 최초 도착 보상의 중복 지급 방지
        return self._get_obs(), info

    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()          # 부모 환경의 4 * H * W 관측
        ch4 = np.zeros((self.H, self.W), dtype=np.float32)
        ch4[:, self.target_lane_col] = 1.0
        return np.concatenate([base, ch4.ravel()])

    def step(self, action: int):
        self.step_count += 1
        reward = REWARD_LANE_STEP

        if action in MOVE_DELTA:
            reward += self._handle_move(action)
        elif action == ACT_SCOUT:
            reward += self._handle_scout()
        elif action == ACT_HARVEST:
            reward += self._handle_harvest()
        elif action == ACT_PEST:
            reward += self._handle_pest()

        # 목표 레인 열에 처음 도착했을 때만 도달 보상을 지급한다.
        if not self._goal_reached and self.agent_pos[1] == self.target_lane_col:
            reward += REWARD_GOAL_REACH
            self._goal_reached = True

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
