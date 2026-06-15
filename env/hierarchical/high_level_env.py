from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.hierarchical.lane_executor_env import LaneExecutorEnv
from env.constants import DONE_STATES, REWARD_HL_LANE_DONE, REWARD_HL_ALL_DONE, HL_STEP_COST


class HighLevelFarmEnv(gym.Env):
    """
    Step 2·3의 상위 정책이 다음 목표 레인을 선택하는 메타 환경.

    상위 환경의 step() 한 번은 하위 정책이 목표 레인을 완료하거나 제한
    스텝에 도달할 때까지 실행한다. 내부 LaneExecutorEnv 상태는 상위 스텝
    사이에도 유지되므로 이전 레인에서 처리한 작물은 완료 상태로 남는다.

    관측: 레인별 작업 완료율, 선택적으로 각 레인까지의 정규화 거리 추가
    행동: 다음에 방문할 레인 인덱스(Discrete(n_lanes))
    """

    def __init__(
        self,
        low_level_model,
        n_beds: int = 4,
        field_height: int = 8,
        max_lane_visits: int | None = None,
        include_distances: bool = True,
    ):
        super().__init__()
        self.low_level_model = low_level_model
        self.inner = LaneExecutorEnv(n_beds=n_beds, field_height=field_height)
        self.lane_cols: list[int] = self.inner.lane_cols
        self.n_lanes: int = len(self.lane_cols)
        self.max_lane_visits: int = max_lane_visits or self.n_lanes * 3
        self.include_distances = include_distances

        obs_size = self.n_lanes * (2 if include_distances else 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_lanes)
        self._lane_visits: int = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.inner.reset(seed=seed)
        self._lane_visits = 0
        return self._get_hl_obs(), {}

    def _get_hl_obs(self) -> np.ndarray:
        """
        레인별 완료율과 선택적인 거리 정보를 상위 관측으로 반환한다.

        [lane0_done%, ..., lane4_done%,         # 완료율
         dist_to_lane0, ..., dist_to_lane4]     # 정규화 거리

        가까운 미완료 레인은 완료율과 거리 값이 모두 낮다.
        """
        completion = np.zeros(self.n_lanes, dtype=np.float32)
        distances = np.zeros(self.n_lanes, dtype=np.float32)
        agent_col = float(self.inner.agent_pos[1])
        max_dist = float(self.inner.W - 1)

        for i, lane_col in enumerate(self.lane_cols):
            crops = self.inner._adjacent_lane_crops(lane_col)
            if crops:
                done = sum(1 for (r, c) in crops
                           if self.inner.crop_states[r, c] in DONE_STATES)
                completion[i] = done / len(crops)
            if self.include_distances:
                distances[i] = abs(agent_col - lane_col) / max_dist

        if self.include_distances:
            return np.concatenate([completion, distances])
        return completion

    def _is_lane_already_done(self, lane_col: int) -> bool:
        """해당 레인에 인접한 작물이 이미 모두 완료되었는지 확인한다."""
        crops = self.inner._adjacent_lane_crops(lane_col)
        return bool(crops) and all(
            self.inner.crop_states[r, c] in DONE_STATES for (r, c) in crops
        )

    def step(self, action: int):
        target_col = self.lane_cols[action]
        was_already_done = self._is_lane_already_done(target_col)

        self.inner.target_lane_col = target_col
        self.inner.step_count = 0
        self.inner._goal_reached = False   # 새 레인 방문마다 최초 도달 보상을 다시 허용

        ll_obs = self.inner._get_obs()
        lane_done = lane_trunc = False
        total_steps = 0
        last_info: dict = {}

        while not (lane_done or lane_trunc):
            ll_action, _ = self.low_level_model.predict(
                ll_obs,
                deterministic=True,
                action_masks=self.inner.action_masks(),
            )
            ll_obs, _, lane_done, lane_trunc, last_info = self.inner.step(int(ll_action))
            total_steps += 1

        self._lane_visits += 1

        hl_reward = -total_steps * HL_STEP_COST
        # 이미 완료한 레인을 재방문했을 때는 완료 보너스를 중복 지급하지 않는다.
        if lane_done and not was_already_done:
            hl_reward += REWARD_HL_LANE_DONE

        all_done = self.inner._is_complete()
        if all_done:
            hl_reward += REWARD_HL_ALL_DONE

        terminated = all_done
        truncated = (self._lane_visits >= self.max_lane_visits) and not terminated

        return (
            self._get_hl_obs(),
            float(hl_reward),
            terminated,
            truncated,
            {
                "steps_for_lane": total_steps,
                "coverage": last_info.get("coverage", 0.0),
                "lane_visits": self._lane_visits,
            },
        )
