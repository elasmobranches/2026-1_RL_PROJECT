"""
계층형 SAC의 하위 정책이 사용하는 연속 목표 레인 환경.

ContinuousFarmEnvCurriculum에 다음 기능을 추가한다.
- target_lane_x: 처리할 목표 레인의 x 좌표
- 관측에 목표 레인의 상대 위치 추가(28차원 → 29차원)
- 목표 레인 인접 작물이 모두 완료되면 종료
- 목표 레인 최초 도착 보너스
"""
from __future__ import annotations
import numpy as np
from gymnasium import spaces
from env.continuous_farm_env_curriculum import ContinuousFarmEnvCurriculum
from env.continuous_farm_env import CELL_SIZE, DONE_STATES, MAX_STEPS

REWARD_GOAL_REACH_CONT = 2.0   # 목표 레인 x 좌표에 처음 도착할 때 지급


class ContinuousLaneExecutorEnv(ContinuousFarmEnvCurriculum):

    def __init__(self, n_beds=3, field_height=5, max_steps=MAX_STEPS, render_mode=None):
        super().__init__(n_beds=n_beds, field_height=field_height,
                         max_steps=max_steps, render_mode=render_mode)
        self.target_lane_x: float = self.lane_x[0]
        self._goal_reached_cont: bool = False

        # 부모 관측 28차원 + 목표 레인 상대 거리 1차원 = 29차원
        parent_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(parent_dim + 1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._goal_reached_cont = False
        return self._get_obs(), info

    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()                              # 부모 환경의 28차원 관측
        dx = np.float32((self.target_lane_x - self.robot_pos[0]) / self.W_m)
        return np.clip(np.append(base, dx).astype(np.float32), -1.0, 1.0)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # 목표 레인에 처음 도착했을 때만 도달 보상을 지급한다.
        if not self._goal_reached_cont:
            dist_to_lane = abs(self.robot_pos[0] - self.target_lane_x)
            if dist_to_lane < 0.8 * CELL_SIZE:
                reward += REWARD_GOAL_REACH_CONT
                self._goal_reached_cont = True

        # 전체 필드가 아니라 목표 레인이 완료되면 종료되도록 부모 조건을 덮어쓴다.
        lane_done = self._is_target_lane_done()
        if lane_done:
            reward += 10.0   # 이산 LaneExecutorEnv와 같은 크기의 레인 완료 보너스
        terminated = lane_done
        truncated = (self.step_count >= self.max_steps) and not lane_done

        info['lane_coverage'] = self._target_lane_coverage()
        return self._get_obs(), float(reward), terminated, truncated, info

    def _target_lane_crops(self) -> list[int]:
        """목표 레인 x 좌표에서 ±1.5셀 안에 있는 작물 인덱스를 반환한다."""
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
