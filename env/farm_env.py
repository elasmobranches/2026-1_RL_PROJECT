from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.constants import (
    CELL_PATH, CELL_CROP, CELL_WALL,
    STATE_UNKNOWN, STATE_NORMAL_DONE,
    STATE_HARVEST_PENDING, STATE_HARVEST_DONE,
    STATE_PEST_PENDING, STATE_PEST_DONE,
    DONE_STATES,
    N_ACTIONS, MOVE_DELTA,
    ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT, ACT_SCOUT, ACT_HARVEST, ACT_PEST,
    REWARD_STEP, REWARD_COLLISION, REWARD_SCOUT_NEW, REWARD_NORMAL_CONFIRM,
    REWARD_HARVEST, REWARD_PEST, REWARD_COMPLETION,
)
from env.map_generator import generate_field_map, init_crop_states


class FarmEnv(gym.Env):
    """
    Step 1에서 사용하는 이산 격자 온실 환경.

    로봇은 주행 레인만 이동하며, 인접 작물을 예찰한 뒤 공개된 상태에 따라
    수확 또는 방제를 수행한다. 실제 작물 상태는 예찰 전까지 숨겨져 있어
    부분 관측 문제를 구성한다.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_beds: int = 4,
        field_height: int = 8,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.n_beds = n_beds
        self.field_height = field_height
        self.render_mode = render_mode

        self.layout = generate_field_map(n_beds, field_height)
        self.H, self.W = self.layout.shape
        self.max_steps = max_steps or self.H * self.W * 3

        self._crop_cells = [tuple(pos) for pos in np.argwhere(self.layout == CELL_CROP)]
        self._path_cells = [tuple(pos) for pos in np.argwhere(self.layout == CELL_PATH)]

        obs_dim = 4 * self.H * self.W
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        # 에피소드마다 reset()에서 다시 초기화되는 상태
        self.agent_pos: tuple[int, int] = (1, 1)
        self._true_states: np.ndarray = np.zeros((self.H, self.W), dtype=np.int32)
        self.crop_states: np.ndarray = np.zeros((self.H, self.W), dtype=np.int32)
        self.step_count: int = 0
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agent_pos = (1, 1)          # 상단 헤드랜드의 첫 번째 레인
        self._true_states = init_crop_states(self.layout, self._rng)
        self.crop_states = np.zeros((self.H, self.W), dtype=np.int32)
        self.step_count = 0

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """맵·로봇 위치·예찰 여부·공개 작물 상태를 4채널 벡터로 반환한다."""
        ch0 = self.layout.astype(np.float32) / 2.0
        ch1 = np.zeros((self.H, self.W), dtype=np.float32)
        ch1[self.agent_pos] = 1.0
        ch2 = (self.crop_states != STATE_UNKNOWN).astype(np.float32)
        ch3 = self.crop_states.astype(np.float32) / 5.0
        return np.concatenate([ch0.ravel(), ch1.ravel(), ch2.ravel(), ch3.ravel()])

    def _adjacent_crop_cells(self) -> list[tuple[int, int]]:
        """현재 로봇 위치의 상하좌우에 인접한 작물 셀을 반환한다."""
        r, c = self.agent_pos
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W and self.layout[nr, nc] == CELL_CROP:
                result.append((nr, nc))
        return result

    def action_masks(self) -> np.ndarray:
        """현재 상태에서 실행 가능한 행동만 MaskablePPO에 전달한다."""
        masks = np.zeros(N_ACTIONS, dtype=bool)
        r, c = self.agent_pos

        for act, (dr, dc) in MOVE_DELTA.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W and self.layout[nr, nc] == CELL_PATH:
                masks[act] = True

        adj = self._adjacent_crop_cells()
        masks[ACT_SCOUT]   = any(self.crop_states[p] == STATE_UNKNOWN for p in adj)
        masks[ACT_HARVEST] = any(self.crop_states[p] == STATE_HARVEST_PENDING for p in adj)
        masks[ACT_PEST]    = any(self.crop_states[p] == STATE_PEST_PENDING for p in adj)

        return masks

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

        terminated = self._is_complete()
        if terminated:
            reward += REWARD_COMPLETION

        truncated = (self.step_count >= self.max_steps) and not terminated

        if self.render_mode == "human":
            self.render()

        info = {
            "coverage": self._coverage_rate(),
            "step": self.step_count,
            "terminated": terminated,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _handle_move(self, action: int) -> float:
        dr, dc = MOVE_DELTA[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.H and 0 <= nc < self.W and self.layout[nr, nc] == CELL_PATH:
            self.agent_pos = (nr, nc)
            return 0.0
        return REWARD_COLLISION

    def _handle_scout(self) -> float:
        """인접한 미예찰 작물의 실제 상태를 공개하고 예찰 보상을 계산한다."""
        total = 0.0
        for pos in self._adjacent_crop_cells():
            if self.crop_states[pos] == STATE_UNKNOWN:
                self.crop_states[pos] = self._true_states[pos]
                total += REWARD_SCOUT_NEW
                if self.crop_states[pos] == STATE_NORMAL_DONE:
                    total += REWARD_NORMAL_CONFIRM
        return total

    def _handle_harvest(self) -> float:
        total = 0.0
        for pos in self._adjacent_crop_cells():
            if self.crop_states[pos] == STATE_HARVEST_PENDING:
                self.crop_states[pos] = STATE_HARVEST_DONE
                total += REWARD_HARVEST
        return total

    def _handle_pest(self) -> float:
        total = 0.0
        for pos in self._adjacent_crop_cells():
            if self.crop_states[pos] == STATE_PEST_PENDING:
                self.crop_states[pos] = STATE_PEST_DONE
                total += REWARD_PEST
        return total

    def _is_complete(self) -> bool:
        """모든 작물이 정상 확인·수확 완료·방제 완료 상태인지 확인한다."""
        return all(self.crop_states[p] in DONE_STATES for p in self._crop_cells)

    def _coverage_rate(self) -> float:
        if not self._crop_cells:
            return 1.0
        done = sum(1 for p in self._crop_cells if self.crop_states[p] in DONE_STATES)
        return done / len(self._crop_cells)

    def render(self):
        symbols = {CELL_PATH: ".", CELL_CROP: "C", CELL_WALL: "#"}
        state_sym = {
            STATE_UNKNOWN: "?", STATE_NORMAL_DONE: "N",
            STATE_HARVEST_PENDING: "H", STATE_PEST_PENDING: "P",
            STATE_HARVEST_DONE: "h", STATE_PEST_DONE: "p",
        }
        r_a, c_a = self.agent_pos
        rows = []
        for r in range(self.H):
            row = []
            for c in range(self.W):
                if (r, c) == (r_a, c_a):
                    row.append("A")
                elif self.layout[r, c] == CELL_CROP:
                    row.append(state_sym[self.crop_states[r, c]])
                else:
                    row.append(symbols[self.layout[r, c]])
            rows.append(" ".join(row))
        print("\n".join(rows))
        print(f"Step: {self.step_count} | Coverage: {self._coverage_rate():.1%}")
        print()
