import numpy as np
from env.constants import (
    CELL_PATH, CELL_CROP, CELL_WALL,
    STATE_UNKNOWN, CROP_STATE_PROBS, CROP_STATE_VALUES
)


def generate_field_map(n_lanes: int = 3, field_height: int = 6) -> np.ndarray:
    """
    Row-based field map with vertical driving lanes.

    H = field_height + 4  (2 walls + 2 headlands)
    W = 2*n_lanes + 3     (walls + alternating C/P cols + outer crop col)

    Field column pattern (col 1..W-2): C P C P C P C
      odd cols  → CELL_CROP
      even cols → CELL_PATH (driving lanes)
    """
    H = field_height + 4
    W = 2 * n_lanes + 3
    layout = np.full((H, W), CELL_WALL, dtype=np.int32)

    layout[1, 1:-1] = CELL_PATH       # top headland
    layout[-2, 1:-1] = CELL_PATH      # bottom headland

    for row in range(2, H - 2):       # field rows
        for col in range(1, W - 1):
            layout[row, col] = CELL_CROP if col % 2 == 1 else CELL_PATH

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
