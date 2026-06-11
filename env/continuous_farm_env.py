"""
ContinuousFarmEnv — Step 4: continuous 2D space + proximity-based interactions.

The robot moves in continuous (x, y) coordinates. Scouting and harvesting/pest-control
happen automatically when the robot is within the respective radius of a crop cell.
This removes discrete Scout/Harvest/Pest actions and enables pure continuous control → SAC.

Layout mirrors the grid env: crop beds separated by driving lanes.
  x-axis: column direction (0 .. W_m)
  y-axis: row direction   (0 .. H_m)
Each grid cell is CELL_SIZE metres wide/tall.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

CELL_SIZE    = 1.0   # metres per grid cell
SCOUT_RADIUS = 0.9   # auto-scout when within this distance of crop centre
ACT_RADIUS   = 0.7   # auto-harvest/pest-control when within this distance
MAX_SPEED    = 0.4   # max displacement per step (in cell units)
MAX_STEPS    = 1200  # episode step limit

# Crop states (same semantics as grid env)
STATE_UNKNOWN         = 0
STATE_NORMAL_DONE     = 1
STATE_HARVEST_PENDING = 2
STATE_PEST_PENDING    = 3
STATE_HARVEST_DONE    = 4
STATE_PEST_DONE       = 5
DONE_STATES = {STATE_NORMAL_DONE, STATE_HARVEST_DONE, STATE_PEST_DONE}

CROP_STATE_PROBS  = [0.60, 0.25, 0.15]
CROP_STATE_VALUES = [STATE_NORMAL_DONE, STATE_HARVEST_PENDING, STATE_PEST_PENDING]

# Rewards
R_STEP          = -0.05
R_SCOUT_NEW     =  1.5
R_NORMAL        =  0.5
R_HARVEST       = 10.0
R_PEST          =  8.0
R_COMPLETION    = 20.0
R_COLLISION     = -1.0
R_PROXIMITY     =  0.3   # shaping: reward per step for being near unprocessed crop


class ContinuousFarmEnv(gym.Env):
    """
    Continuous 2-D farm environment for SAC.

    Map: same P/CC lane structure as grid env, but coordinates are real-valued.
    Action: (vx, vy) ∈ [−1, 1]² → clipped to MAX_SPEED and applied as displacement.
    Interactions: proximity-based (no explicit discrete actions).

    Observation (flat float32):
      [rx, ry,                      # robot position normalised to [-1, 1]
       cos(heading), sin(heading),  # approximate heading from last move
       per-crop: dx, dy, revealed, state/5]  # relative pos + state for each crop
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

        # Grid dimensions (same formula as grid env)
        self.G_H = field_height + 4          # grid rows
        self.G_W = 3 * n_beds + 3            # grid cols
        self.W_m = self.G_W * CELL_SIZE      # continuous width
        self.H_m = self.G_H * CELL_SIZE      # continuous height

        # Lane cols (driving lanes)
        self.lane_x = [
            (c + 0.5) * CELL_SIZE
            for c in range(1, self.G_W - 1)
            if (c - 1) % 3 == 0
        ]

        # Crop cell centres
        self.crop_centres: list[tuple[float, float]] = []
        for row in range(2, self.G_H - 2):
            for col in range(1, self.G_W - 1):
                if (col - 1) % 3 != 0:   # crop cols: (col-1)%3 in {1,2}
                    cx = (col + 0.5) * CELL_SIZE
                    cy = (row + 0.5) * CELL_SIZE
                    self.crop_centres.append((cx, cy))
        self.n_crops = len(self.crop_centres)

        # Spaces
        # obs: [rx, ry, cos_h, sin_h, (dx, dy, rev, state)*n_crops]
        obs_dim = 4 + self.n_crops * 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Mutable state
        self.robot_pos   = np.zeros(2, dtype=np.float64)
        self._last_move  = np.array([0.0, 1.0])
        self._true_states  = np.zeros(self.n_crops, dtype=np.int32)
        self.crop_states   = np.zeros(self.n_crops, dtype=np.int32)
        self.step_count  = 0
        self._rng = np.random.default_rng()

    # ──────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Start at top headland, first lane x
        # Randomize start lane each episode so agent must generalise navigation
        lane_idx = int(self._rng.integers(len(self.lane_x)))
        self.robot_pos = np.array([self.lane_x[lane_idx], 1.5 * CELL_SIZE])
        self._last_move = np.array([0.0, 1.0])
        self._true_states = self._rng.choice(
            CROP_STATE_VALUES, size=self.n_crops, p=CROP_STATE_PROBS
        ).astype(np.int32)
        self.crop_states = np.zeros(self.n_crops, dtype=np.int32)
        self.step_count = 0
        self._prev_potential = self._potential()  # for shaping

        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        self.step_count += 1
        reward = R_STEP

        # Move
        vel = np.clip(action, -1.0, 1.0) * MAX_SPEED * CELL_SIZE
        new_pos = self.robot_pos + vel

        # Soft boundary — stay inside drivable area (headlands + lanes)
        new_pos[0] = np.clip(new_pos[0], 0.5 * CELL_SIZE, (self.G_W - 0.5) * CELL_SIZE)
        new_pos[1] = np.clip(new_pos[1], 0.5 * CELL_SIZE, (self.G_H - 0.5) * CELL_SIZE)

        # Collision with crop beds — penalty, no movement into crop
        if self._in_crop_bed(new_pos):
            reward += R_COLLISION
            new_pos = self.robot_pos.copy()  # block movement
        else:
            self.robot_pos = new_pos
            if np.linalg.norm(vel) > 1e-6:
                self._last_move = vel / (np.linalg.norm(vel) + 1e-8)

        # Proximity-based interactions
        reward += self._interact()

        # Potential-based shaping: F = γ·φ(s') − φ(s)
        # Encourages approaching crops WITHOUT providing a standing reward trap
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

    # ──────────────────────────────────────────────────────────────────
    def _interact(self) -> float:
        """Auto-scout and auto-harvest/pest within proximity radius."""
        rx, ry = self.robot_pos
        total = 0.0
        for i, (cx, cy) in enumerate(self.crop_centres):
            dist = np.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)

            # Scout: reveal hidden crop
            if dist <= SCOUT_RADIUS and self.crop_states[i] == STATE_UNKNOWN:
                self.crop_states[i] = self._true_states[i]
                total += R_SCOUT_NEW
                if self.crop_states[i] == STATE_NORMAL_DONE:
                    total += R_NORMAL

            # Harvest
            if dist <= ACT_RADIUS and self.crop_states[i] == STATE_HARVEST_PENDING:
                self.crop_states[i] = STATE_HARVEST_DONE
                total += R_HARVEST

            # Pest control
            if dist <= ACT_RADIUS and self.crop_states[i] == STATE_PEST_PENDING:
                self.crop_states[i] = STATE_PEST_DONE
                total += R_PEST

        return total

    # ──────────────────────────────────────────────────────────────────
    def _in_crop_bed(self, pos: np.ndarray) -> bool:
        """
        True if position is inside a non-drivable cell.
        Blocked: outer walls (col 0, col G_W-1, row 0, row G_H-1)
                 and crop columns in field rows.
        Passable: headlands (row 1, row G_H-2) and driving lane columns.
        """
        px, py = pos
        col = int(px / CELL_SIZE)
        row = int(py / CELL_SIZE)
        col = max(0, min(col, self.G_W - 1))
        row = max(0, min(row, self.G_H - 1))

        # Outer wall cells are blocked
        if col == 0 or col == self.G_W - 1:
            return True
        if row == 0 or row == self.G_H - 1:
            return True

        # Headland rows (row 1, row G_H-2) are fully passable driving areas
        if row == 1 or row == self.G_H - 2:
            return False

        # Field rows: lane cols are passable, crop cols are blocked
        inner_col = col - 1            # inner index (1-indexed col → 0-indexed)
        return inner_col % 3 != 0      # lane: (col-1)%3==0 → passable

    # ──────────────────────────────────────────────────────────────────
    def _potential(self) -> float:
        """φ(s) = −min_dist to nearest unprocessed crop (negative distance as potential)."""
        rx, ry = self.robot_pos
        unprocessed = [
            (cx, cy) for i, (cx, cy) in enumerate(self.crop_centres)
            if self.crop_states[i] not in DONE_STATES
        ]
        if not unprocessed:
            return 0.0
        min_dist = min(np.sqrt((rx - cx) ** 2 + (ry - cy) ** 2) for cx, cy in unprocessed)
        return -min_dist * R_PROXIMITY   # negative: closer = higher potential

    # ──────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        rx, ry = self.robot_pos
        nx = rx / self.W_m * 2 - 1   # normalise to [−1, 1]
        ny = ry / self.H_m * 2 - 1
        cos_h, sin_h = float(self._last_move[0]), float(self._last_move[1])

        crop_feats = []
        for i, (cx, cy) in enumerate(self.crop_centres):
            dx = (cx - rx) / self.W_m    # relative position normalised
            dy = (cy - ry) / self.H_m
            rev = 1.0 if self.crop_states[i] != STATE_UNKNOWN else 0.0
            sv  = self.crop_states[i] / 5.0
            crop_feats.extend([dx, dy, rev, sv])

        obs = np.array([nx, ny, cos_h, sin_h] + crop_feats, dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _is_complete(self) -> bool:
        return all(self.crop_states[i] in DONE_STATES for i in range(self.n_crops))

    def _coverage_rate(self) -> float:
        done = sum(1 for s in self.crop_states if s in DONE_STATES)
        return done / self.n_crops if self.n_crops > 0 else 1.0

    def render(self):
        print(f"pos=({self.robot_pos[0]:.2f},{self.robot_pos[1]:.2f}) "
              f"coverage={self._coverage_rate():.0%} step={self.step_count}")
