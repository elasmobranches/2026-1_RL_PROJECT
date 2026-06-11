from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.hierarchical.lane_executor_env import LaneExecutorEnv
from env.constants import DONE_STATES, REWARD_HL_LANE_DONE, REWARD_HL_ALL_DONE, HL_STEP_COST


class HighLevelFarmEnv(gym.Env):
    """
    High-level env: select which lane to visit next.

    Each step() executes the low-level policy until target lane is complete
    (or max_steps_per_lane exceeded). The inner LaneExecutorEnv state persists
    across high-level steps — crops processed in earlier lanes stay processed.

    Observation: fraction of adjacent crops done per lane (n_lanes-dim float32)
    Action:      lane index to visit next (Discrete(n_lanes))
    """

    def __init__(
        self,
        low_level_model,
        n_beds: int = 4,
        field_height: int = 8,
        max_lane_visits: int | None = None,
    ):
        super().__init__()
        self.low_level_model = low_level_model
        self.inner = LaneExecutorEnv(n_beds=n_beds, field_height=field_height)
        self.lane_cols: list[int] = self.inner.lane_cols
        self.n_lanes: int = len(self.lane_cols)
        self.max_lane_visits: int = max_lane_visits or self.n_lanes * 3

        # obs = [lane_done_rates(n_lanes), normalized_distances(n_lanes)]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_lanes * 2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_lanes)
        self._lane_visits: int = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.inner.reset(seed=seed)
        self._lane_visits = 0
        return self._get_hl_obs(), {}

    def _get_hl_obs(self) -> np.ndarray:
        """
        Returns 2*n_lanes observation:
          [lane0_done%, ..., lane4_done%,        -- completion rates
           dist_to_lane0, ..., dist_to_lane4]    -- normalised distances
        Nearest unfinished lane will have low distance AND low done%.
        """
        completion = np.zeros(self.n_lanes, dtype=np.float32)
        distances  = np.zeros(self.n_lanes, dtype=np.float32)
        agent_col  = float(self.inner.agent_pos[1])
        max_dist   = float(self.inner.W - 1)

        for i, lane_col in enumerate(self.lane_cols):
            crops = self.inner._adjacent_lane_crops(lane_col)
            if crops:
                done = sum(1 for (r, c) in crops
                           if self.inner.crop_states[r, c] in DONE_STATES)
                completion[i] = done / len(crops)
            distances[i] = abs(agent_col - lane_col) / max_dist

        return np.concatenate([completion, distances])

    def _is_lane_already_done(self, lane_col: int) -> bool:
        """True if all crops adjacent to lane_col are already in DONE_STATES."""
        crops = self.inner._adjacent_lane_crops(lane_col)
        return bool(crops) and all(
            self.inner.crop_states[r, c] in DONE_STATES for (r, c) in crops
        )

    def step(self, action: int):
        target_col = self.lane_cols[action]
        was_already_done = self._is_lane_already_done(target_col)

        self.inner.target_lane_col = target_col
        self.inner.step_count = 0

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
        # Only reward for newly completed lane (not revisiting an already-done lane)
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
