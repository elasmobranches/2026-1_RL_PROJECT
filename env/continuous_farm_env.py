"""
ContinuousFarmEnv - Step 4의 연속 2차원 온실 환경.

로봇은 연속 (x, y) 좌표에서 이동한다. 작물과 일정 거리 안에 들어오면
예찰·수확·방제가 자동 수행되므로 별도의 작업 행동 없이 속도 벡터만으로
정책을 학습할 수 있다.

맵 구조는 이산 환경과 동일하게 주행 레인 사이에 재배단이 배치된다.
각 격자 셀의 가로·세로 길이는 CELL_SIZE 미터다.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

CELL_SIZE    = 1.0   # 격자 셀 한 변의 길이(m)
SCOUT_RADIUS = 0.9   # 작물 중심과 이 거리 이내이면 자동 예찰
ACT_RADIUS   = 0.7   # 이 거리 이내이면 자동 수확·방제
MAX_SPEED    = 0.4   # 스텝당 최대 이동 거리(셀 단위)
MAX_STEPS    = 1200  # 에피소드 최대 스텝

# 작물 상태는 이산 환경과 같은 의미를 사용한다.
STATE_UNKNOWN         = 0
STATE_NORMAL_DONE     = 1
STATE_HARVEST_PENDING = 2
STATE_PEST_PENDING    = 3
STATE_HARVEST_DONE    = 4
STATE_PEST_DONE       = 5
DONE_STATES = {STATE_NORMAL_DONE, STATE_HARVEST_DONE, STATE_PEST_DONE}

CROP_STATE_PROBS  = [0.60, 0.25, 0.15]
CROP_STATE_VALUES = [STATE_NORMAL_DONE, STATE_HARVEST_PENDING, STATE_PEST_PENDING]

# 연속 환경 보상
R_STEP          = -0.05
R_SCOUT_NEW     =  1.5
R_NORMAL        =  0.5
R_HARVEST       = 10.0
R_PEST          =  8.0
R_COMPLETION    = 20.0
R_COLLISION     = -1.0
R_PROXIMITY     =  0.3   # 미처리 작물까지의 거리를 사용하는 shaping 계수


class ContinuousFarmEnv(gym.Env):
    """
    SAC 학습을 위한 연속 2차원 온실 환경.

    맵은 이산 환경과 같은 레인/재배단 구조지만 좌표를 실수로 표현한다.
    행동 (vx, vy)는 [-1, 1] 범위에서 MAX_SPEED만큼 스케일링해 이동량으로
    사용하며, 작물 작업은 거리 기반으로 자동 수행한다.

    관측(flat float32):
      [rx, ry,                       # [-1, 1]로 정규화한 로봇 위치
       cos(heading), sin(heading),   # 마지막 이동으로 계산한 진행 방향
       per-crop: dx, dy, revealed, state/5]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_beds: int = 3,
        field_height: int = 5,
        max_steps: int = MAX_STEPS,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.n_beds      = n_beds
        self.field_height = field_height
        self.max_steps   = max_steps
        self.render_mode = render_mode

        # 이산 환경과 같은 공식으로 격자 크기를 계산한 뒤 연속 좌표로 변환한다.
        self.G_H = field_height + 4
        self.G_W = 3 * n_beds + 3
        self.W_m = self.G_W * CELL_SIZE
        self.H_m = self.G_H * CELL_SIZE

        # 주행 레인 중심의 연속 x 좌표
        self.lane_x = [
            (c + 0.5) * CELL_SIZE
            for c in range(1, self.G_W - 1)
            if (c - 1) % 3 == 0
        ]

        # 거리 계산을 벡터화하기 위해 작물 중심 좌표를 배열로 보관한다.
        centres = []
        for row in range(2, self.G_H - 2):
            for col in range(1, self.G_W - 1):
                if (col - 1) % 3 != 0:
                    centres.append([(col + 0.5) * CELL_SIZE, (row + 0.5) * CELL_SIZE])
        self._crop_arr = np.array(centres, dtype=np.float64)   # (n_crops, 2)
        self.crop_centres = [tuple(c) for c in self._crop_arr]  # 렌더링 코드 호환용
        self.n_crops = len(self._crop_arr)

        # 관측: [rx, ry, cos_h, sin_h, (dx, dy, rev, state)*n_crops]
        obs_dim = 4 + self.n_crops * 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # 에피소드마다 변하는 상태
        self.robot_pos   = np.zeros(2, dtype=np.float64)
        self._last_move  = np.array([0.0, 1.0])
        self._true_states  = np.zeros(self.n_crops, dtype=np.int32)
        self.crop_states   = np.zeros(self.n_crops, dtype=np.int32)
        self.step_count  = 0
        self._prev_potential: float = 0.0   # reset() 전 step() 호출에도 안전하도록 초기화
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # 상단 헤드랜드의 임의 레인에서 시작해 특정 시작점에 과적합하지 않게 한다.
        lane_idx = int(self._rng.integers(len(self.lane_x)))
        self.robot_pos = np.array([self.lane_x[lane_idx], 1.5 * CELL_SIZE])
        self._last_move = np.array([0.0, 1.0])
        self._true_states = self._rng.choice(
            CROP_STATE_VALUES, size=self.n_crops, p=CROP_STATE_PROBS
        ).astype(np.int32)
        self.crop_states = np.zeros(self.n_crops, dtype=np.int32)
        self.step_count = 0
        self._prev_potential = self._potential()  # potential shaping 기준값

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        reward = R_STEP

        # 행동을 실제 이동량으로 변환한다.
        vel = np.clip(action, -1.0, 1.0) * MAX_SPEED * CELL_SIZE
        new_pos = self.robot_pos + vel

        # 좌표가 환경 외부로 벗어나지 않도록 경계 안으로 제한한다.
        new_pos[0] = np.clip(new_pos[0], 0.5 * CELL_SIZE, (self.G_W - 0.5) * CELL_SIZE)
        new_pos[1] = np.clip(new_pos[1], 0.5 * CELL_SIZE, (self.G_H - 0.5) * CELL_SIZE)

        # 재배단과 충돌하면 페널티를 주고 이동을 취소한다.
        if self._in_crop_bed(new_pos):
            reward += R_COLLISION
            new_pos = self.robot_pos.copy()
        else:
            self.robot_pos = new_pos
            if np.linalg.norm(vel) > 1e-6:
                self._last_move = vel / (np.linalg.norm(vel) + 1e-8)

        # 거리 기반 자동 예찰·수확·방제
        reward += self._interact()

        # Potential-based shaping은 한곳에 머물며 보상을 반복 획득하는 문제 없이
        # 미처리 작물에 가까워지는 행동을 유도한다.
        new_potential = self._potential()
        reward += 0.99 * new_potential - self._prev_potential
        self._prev_potential = new_potential

        terminated = self._is_complete()
        if terminated:
            reward += R_COMPLETION
        truncated = (self.step_count >= self.max_steps) and not terminated

        return self._get_obs(), float(reward), terminated, truncated, {
            "coverage": self._coverage_rate(),
            "step": self.step_count,
        }

    def _interact(self) -> float:
        """반경 안의 작물을 자동 예찰·수확·방제하고 보상을 계산한다."""
        dists = np.linalg.norm(self._crop_arr - self.robot_pos, axis=1)
        total = 0.0

        # 예찰 반경 안의 미지 작물 상태를 공개한다.
        scout_mask = (dists <= SCOUT_RADIUS) & (self.crop_states == STATE_UNKNOWN)
        if scout_mask.any():
            self.crop_states[scout_mask] = self._true_states[scout_mask]
            newly_normal = scout_mask & (self.crop_states == STATE_NORMAL_DONE)
            total += scout_mask.sum() * R_SCOUT_NEW
            total += newly_normal.sum() * R_NORMAL

        # 작업 반경 안의 수확 대기 작물을 완료 상태로 바꾼다.
        harvest_mask = (dists <= ACT_RADIUS) & (self.crop_states == STATE_HARVEST_PENDING)
        if harvest_mask.any():
            self.crop_states[harvest_mask] = STATE_HARVEST_DONE
            total += harvest_mask.sum() * R_HARVEST

        # 작업 반경 안의 방제 대기 작물을 완료 상태로 바꾼다.
        pest_mask = (dists <= ACT_RADIUS) & (self.crop_states == STATE_PEST_PENDING)
        if pest_mask.any():
            self.crop_states[pest_mask] = STATE_PEST_DONE
            total += pest_mask.sum() * R_PEST

        return total

    def _in_crop_bed(self, pos: np.ndarray) -> bool:
        """
        위치가 주행 불가능한 셀 내부인지 확인한다.

        외곽 벽과 필드 행의 작물 열은 막혀 있고, 헤드랜드와 주행 레인은
        통과할 수 있다.
        """
        px, py = pos
        col = int(px / CELL_SIZE)
        row = int(py / CELL_SIZE)
        col = max(0, min(col, self.G_W - 1))
        row = max(0, min(row, self.G_H - 1))

        # 외곽 벽은 항상 통과할 수 없다.
        if col == 0 or col == self.G_W - 1:
            return True
        if row == 0 or row == self.G_H - 1:
            return True

        # 상단·하단 헤드랜드는 레인 사이를 이동하는 통로다.
        if row == 1 or row == self.G_H - 2:
            return False

        # 필드 행에서는 레인 열만 통과할 수 있다.
        inner_col = col - 1
        return inner_col % 3 != 0

    def _potential(self) -> float:
        """가장 가까운 미처리 작물까지의 거리로 잠재 함수 φ(s)를 계산한다."""
        unprocessed_mask = ~np.isin(self.crop_states, list(DONE_STATES))
        if not unprocessed_mask.any():
            return 0.0
        dists = np.linalg.norm(self._crop_arr[unprocessed_mask] - self.robot_pos, axis=1)
        return -float(dists.min()) * R_PROXIMITY

    def _get_obs(self) -> np.ndarray:
        rx, ry = self.robot_pos
        nx = rx / self.W_m * 2 - 1   # [-1, 1] 범위로 정규화
        ny = ry / self.H_m * 2 - 1
        cos_h, sin_h = float(self._last_move[0]), float(self._last_move[1])

        crop_feats = []
        for i, (cx, cy) in enumerate(self.crop_centres):
            dx = (cx - rx) / self.W_m    # 정규화한 상대 위치
            dy = (cy - ry) / self.H_m
            rev = 1.0 if self.crop_states[i] != STATE_UNKNOWN else 0.0
            sv  = self.crop_states[i] / 5.0
            crop_feats.extend([dx, dy, rev, sv])

        obs = np.array([nx, ny, cos_h, sin_h] + crop_feats, dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _is_complete(self) -> bool:
        return bool(np.isin(self.crop_states, list(DONE_STATES)).all())

    def _coverage_rate(self) -> float:
        if self.n_crops == 0:
            return 1.0
        return float(np.isin(self.crop_states, list(DONE_STATES)).sum()) / self.n_crops

    def render(self):
        print(f"pos=({self.robot_pos[0]:.2f},{self.robot_pos[1]:.2f}) "
              f"coverage={self._coverage_rate():.0%} step={self.step_count}")
