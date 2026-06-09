import numpy as np
from env.constants import (
    CELL_PATH, CELL_CROP, CELL_WALL,
    STATE_UNKNOWN, CROP_STATE_PROBS, CROP_STATE_VALUES
)


def generate_field_map(n_beds: int = 4, field_height: int = 8) -> np.ndarray:
    """
    Field map with 2-column-wide crop beds and vertical driving lanes.

    Each crop bed is 2 columns wide. Driving lanes separate every bed,
    so a robot in a lane can only scout the immediately adjacent (inner)
    crop column — the far (outer) column must be scouted from the other lane.

    H = field_height + 4
    W = 3*n_beds + 3  (wall + [lane + bed(2)] * n_beds + lane + wall)

    Field column pattern (col 1..W-2):
      (col-1) % 3 == 0  → CELL_PATH  (driving lane)
      (col-1) % 3 == 1  → CELL_CROP  (inner crop of bed)
      (col-1) % 3 == 2  → CELL_CROP  (outer crop of bed)

    Example n_beds=4: W P CC P CC P CC P CC P W  (W=15)
    """
    if n_beds < 1 or field_height < 1:
        raise ValueError(
            f"n_beds and field_height must be >= 1, got n_beds={n_beds}, field_height={field_height}"
        )
    H = field_height + 4
    W = 3 * n_beds + 3
    layout = np.full((H, W), CELL_WALL, dtype=np.int32)

    layout[1, 1:-1] = CELL_PATH       # top headland
    layout[-2, 1:-1] = CELL_PATH      # bottom headland

    for row in range(2, H - 2):       # field rows
        for col in range(1, W - 1):
            layout[row, col] = CELL_PATH if (col - 1) % 3 == 0 else CELL_CROP

    return layout


def init_crop_states(layout: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """
    Assign random states to all CELL_CROP cells.
    Non-crop cells remain STATE_UNKNOWN (0).
    """
    if rng is None:
        rng = np.random.default_rng()

    states = np.zeros(layout.shape, dtype=np.int32)
    crop_positions = np.argwhere(layout == CELL_CROP)

    chosen = rng.choice(CROP_STATE_VALUES, size=len(crop_positions), p=CROP_STATE_PROBS)
    for (r, c), state in zip(crop_positions, chosen):
        states[r, c] = state

    return states
