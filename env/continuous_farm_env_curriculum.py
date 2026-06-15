"""
Step 4의 단순화 관측 및 선택적 커리큘럼 환경.

커리큘럼 단계(CurriculumCallback 사용 시 자동 상승):
  0: 임의 작물 바로 옆 레인에서 시작하여 작업 수행을 먼저 학습
  1: 필드 내부의 임의 레인에서 시작하여 필드 주행을 학습
  2: 일반 헤드랜드에서 시작하여 전체 에피소드를 학습

관측은 전체 작물을 포함한 124차원에서 로봇과 가까운 미처리 작물 5개만
포함하는 28차원으로 줄여 정책의 탐색 공간을 단순화한다.
"""
from __future__ import annotations
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from env.continuous_farm_env import ContinuousFarmEnv, CELL_SIZE, DONE_STATES, STATE_UNKNOWN, MAX_STEPS


N_OBS_CROPS = 5   # 관측에 포함할 가까운 미처리 작물 수


class ContinuousFarmEnvCurriculum(ContinuousFarmEnv):
    """ContinuousFarmEnv에 커리큘럼 시작점과 28차원 관측을 추가한 환경."""

    def __init__(
        self,
        n_beds: int = 3,
        field_height: int = 5,
        max_steps: int = MAX_STEPS,
        render_mode: str | None = None,
    ):
        super().__init__(n_beds=n_beds, field_height=field_height,
                         max_steps=max_steps, render_mode=render_mode)
        self.curriculum_level: int = 0

        # 관측 = 로봇 위치(2) + 방향(2) + 이동 가능 방향(4) + 작물 특징(K×4)
        obs_dim = 4 + 4 + N_OBS_CROPS * 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)     # 작물 상태와 난수 생성기 등을 초기화

        if self.curriculum_level == 0:
            self._spawn_near_crop()
        elif self.curriculum_level == 1:
            self._spawn_in_field_lane()
        # 레벨 2 이상에서는 부모 환경의 헤드랜드 시작점을 유지한다.

        self._prev_potential = self._potential()
        return self._get_obs(), info

    def _spawn_near_crop(self):
        """레벨 0: 임의 작물에 바로 인접한 레인에서 시작한다."""
        idx = int(self._rng.integers(self.n_crops))
        cx, cy = self._crop_arr[idx]

        # 선택한 작물과 가장 가까운 주행 레인을 찾는다.
        nearest_lane = min(self.lane_x, key=lambda lx: abs(lx - cx))

        # 시작 y 좌표를 실제 필드 행 안으로 제한한다.
        field_y_min = 2.5 * CELL_SIZE
        field_y_max = (self.G_H - 2.5) * CELL_SIZE
        ry = float(np.clip(cy, field_y_min, field_y_max))

        self.robot_pos = np.array([nearest_lane, ry])

    def _spawn_in_field_lane(self):
        """레벨 1: 필드 내부의 임의 레인과 행에서 시작한다."""
        lane_idx = int(self._rng.integers(len(self.lane_x)))
        row = int(self._rng.integers(2, self.G_H - 2))   # 헤드랜드를 제외한 필드 행
        self.robot_pos = np.array([
            self.lane_x[lane_idx],
            (row + 0.5) * CELL_SIZE,
        ])

    def _get_obs(self) -> np.ndarray:
        """로봇 자세·이동 가능 방향·가까운 작물로 구성된 28차원 관측을 반환한다."""
        rx, ry = self.robot_pos
        nx = rx / self.W_m * 2 - 1
        ny = ry / self.H_m * 2 - 1
        cos_h = float(self._last_move[0])
        sin_h = float(self._last_move[1])

        dists = np.linalg.norm(self._crop_arr - self.robot_pos, axis=1)

        # 미처리 작물을 우선하고 각 집합을 거리순으로 정렬한다.
        unproc = np.where(~np.isin(self.crop_states, list(DONE_STATES)))[0]
        proc   = np.where( np.isin(self.crop_states, list(DONE_STATES)))[0]

        sorted_unproc = unproc[np.argsort(dists[unproc])] if len(unproc) else np.array([], int)
        sorted_proc   = proc  [np.argsort(dists[proc  ])] if len(proc  ) else np.array([], int)

        # K개 슬롯을 미처리 작물, 처리 완료 작물, 더미 순서로 채운다.
        k_indices = list(sorted_unproc[:N_OBS_CROPS])
        if len(k_indices) < N_OBS_CROPS:
            k_indices += list(sorted_proc[:N_OBS_CROPS - len(k_indices)])

        crop_feats: list[float] = []
        for idx in k_indices:
            cx, cy = self._crop_arr[idx]
            crop_feats += [
                (cx - rx) / self.W_m,
                (cy - ry) / self.H_m,
                1.0 if self.crop_states[idx] != STATE_UNKNOWN else 0.0,
                self.crop_states[idx] / 5.0,
            ]
        # 작물이 부족하면 완료 상태를 나타내는 더미 특징으로 채운다.
        while len(crop_feats) < N_OBS_CROPS * 4:
            crop_feats += [0.0, 0.0, 1.0, 0.2]

        # 인접 셀 안쪽까지 검사해 네 방향 이동 가능 여부를 만든다(1=가능, 0=막힘).
        step = CELL_SIZE * 0.6
        nav = [
            0.0 if self._in_crop_bed(self.robot_pos + np.array([ 0,  step])) else 1.0,  # N
            0.0 if self._in_crop_bed(self.robot_pos + np.array([ 0, -step])) else 1.0,  # S
            0.0 if self._in_crop_bed(self.robot_pos + np.array([ step, 0])) else 1.0,  # E
            0.0 if self._in_crop_bed(self.robot_pos + np.array([-step, 0])) else 1.0,  # W
        ]

        obs = np.array([nx, ny, cos_h, sin_h] + nav + crop_feats, dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)


class CurriculumCallback(BaseCallback):
    """
    최근 성공률이 기준을 넘으면 커리큘럼 단계를 자동으로 올린다.

    성공은 에피소드 커버리지가 success_threshold 이상인 경우로 정의한다.
    최근 window개 에피소드 중 성공 비율이 level_up_at 이상이면 다음 단계로
    이동한다.
    """

    def __init__(
        self,
        vec_env,
        success_threshold: float = 0.7,
        window: int = 20,
        level_up_at: float = 0.7,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.success_threshold = success_threshold
        self.window = window
        self.level_up_at = level_up_at
        self._coverages: list[float] = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done:
                self._coverages.append(info.get("coverage", 0.0))

        if len(self._coverages) >= self.window:
            recent = self._coverages[-self.window:]
            rate = sum(1 for c in recent if c >= self.success_threshold) / self.window

            current = self.vec_env.envs[0].unwrapped.curriculum_level
            if rate >= self.level_up_at and current < 2:
                new_level = current + 1
                for env in self.vec_env.envs:
                    env.unwrapped.curriculum_level = new_level
                # 이전 단계의 성공 기록으로 다음 단계까지 연속 상승하지 않도록 비운다.
                self._coverages.clear()
                if self.verbose:
                    print(f"\n>>> Curriculum LEVEL UP: {current} → {new_level} "
                          f"(success={rate:.0%} over last {self.window} eps) <<<\n")

        return True
