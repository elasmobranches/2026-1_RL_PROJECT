"""
ContinuousFarmEnvCurriculum — Step 4 (v2): SAC + Curriculum Learning

Curriculum levels (auto-advanced by CurriculumCallback):
  0: Robot spawns in the lane immediately adjacent to a random unprocessed crop
     → agent learns scout/harvest/pest first
  1: Robot spawns at a random field lane position (anywhere in the field)
     → agent learns to navigate within field
  2: Normal headland start (same as original env)
     → agent learns full episode

Observation simplification:
  Original 124-dim (all 30 crops) → 24-dim (robot + 5 nearest unprocessed crops)
  Dramatically reduces policy search space.
"""
from __future__ import annotations
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from env.continuous_farm_env import ContinuousFarmEnv, CELL_SIZE, DONE_STATES, STATE_UNKNOWN, MAX_STEPS


N_OBS_CROPS = 5   # observe K nearest unprocessed crops


class ContinuousFarmEnvCurriculum(ContinuousFarmEnv):
    """Curriculum-enabled wrapper around ContinuousFarmEnv."""

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

        # obs = robot(2) + heading(2) + nav_flags(4) + K×4 crop features
        # nav_flags: can_move_N, can_move_S, can_move_E, can_move_W (0 or 1)
        obs_dim = 4 + 4 + N_OBS_CROPS * 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    # ──────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)     # resets crop states, rng, etc.

        if self.curriculum_level == 0:
            self._spawn_near_crop()
        elif self.curriculum_level == 1:
            self._spawn_in_field_lane()
        # level 2+: keep headland start from super().reset()

        self._prev_potential = self._potential()
        return self._get_obs(), info

    # ──────────────────────────────────────────────────────────────────
    def _spawn_near_crop(self):
        """Level 0: spawn in the lane directly adjacent to a random crop."""
        idx = int(self._rng.integers(self.n_crops))
        cx, cy = self._crop_arr[idx]

        # Find nearest driving lane x
        nearest_lane = min(self.lane_x, key=lambda lx: abs(lx - cx))

        # Clamp y to field rows
        field_y_min = 2.5 * CELL_SIZE
        field_y_max = (self.G_H - 2.5) * CELL_SIZE
        ry = float(np.clip(cy, field_y_min, field_y_max))

        self.robot_pos = np.array([nearest_lane, ry])

    def _spawn_in_field_lane(self):
        """Level 1: spawn at a random (lane, field-row) position."""
        lane_idx = int(self._rng.integers(len(self.lane_x)))
        row = int(self._rng.integers(2, self.G_H - 2))   # field rows only
        self.robot_pos = np.array([
            self.lane_x[lane_idx],
            (row + 0.5) * CELL_SIZE,
        ])

    # ──────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """24-dim obs: robot pose + K nearest unprocessed crops."""
        rx, ry = self.robot_pos
        nx = rx / self.W_m * 2 - 1
        ny = ry / self.H_m * 2 - 1
        cos_h = float(self._last_move[0])
        sin_h = float(self._last_move[1])

        dists = np.linalg.norm(self._crop_arr - self.robot_pos, axis=1)

        # Prioritise unprocessed crops, sort by distance
        unproc = np.where(~np.isin(self.crop_states, list(DONE_STATES)))[0]
        proc   = np.where( np.isin(self.crop_states, list(DONE_STATES)))[0]

        sorted_unproc = unproc[np.argsort(dists[unproc])] if len(unproc) else np.array([], int)
        sorted_proc   = proc  [np.argsort(dists[proc  ])] if len(proc  ) else np.array([], int)

        # Fill K slots: unprocessed first, then processed, then dummy padding
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
        # Pad with far-away done dummy entries
        while len(crop_feats) < N_OBS_CROPS * 4:
            crop_feats += [0.0, 0.0, 1.0, 0.2]

        # 4-direction navigation flags (1=passable, 0=blocked)
        # step > 0.5*CELL_SIZE to probe into adjacent cell
        step = CELL_SIZE * 0.6
        nav = [
            0.0 if self._in_crop_bed(self.robot_pos + np.array([ 0,  step])) else 1.0,  # N
            0.0 if self._in_crop_bed(self.robot_pos + np.array([ 0, -step])) else 1.0,  # S
            0.0 if self._in_crop_bed(self.robot_pos + np.array([ step, 0])) else 1.0,  # E
            0.0 if self._in_crop_bed(self.robot_pos + np.array([-step, 0])) else 1.0,  # W
        ]

        obs = np.array([nx, ny, cos_h, sin_h] + nav + crop_feats, dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)


# ──────────────────────────────────────────────────────────────────────
class CurriculumCallback(BaseCallback):
    """
    Auto-advances curriculum level when recent success rate exceeds threshold.

    success = episode coverage >= success_threshold (default 0.7 = 70% crops done)
    window  = number of recent episodes to evaluate
    level_up_at = fraction of window that must be successes to advance
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
                if self.verbose:
                    print(f"\n>>> Curriculum LEVEL UP: {current} → {new_level} "
                          f"(success={rate:.0%} over last {self.window} eps) <<<\n")

        return True
