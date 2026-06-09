import numpy as np
import pytest
from env.constants import CELL_PATH, CELL_CROP, CELL_WALL, STATE_UNKNOWN, DONE_STATES, CROP_STATE_VALUES
from env.map_generator import generate_field_map, init_crop_states


def test_map_shape():
    layout = generate_field_map(n_lanes=3, field_height=4)
    expected_H = 4 + 4   # field_height + 2 walls + 2 headlands
    expected_W = 2*3 + 3  # 2*n_lanes + 3
    assert layout.shape == (expected_H, expected_W)


def test_walls_on_border():
    layout = generate_field_map(n_lanes=3, field_height=4)
    H, W = layout.shape
    assert np.all(layout[0, :] == CELL_WALL)
    assert np.all(layout[-1, :] == CELL_WALL)
    assert np.all(layout[:, 0] == CELL_WALL)
    assert np.all(layout[:, -1] == CELL_WALL)


def test_headland_rows_are_path():
    layout = generate_field_map(n_lanes=3, field_height=4)
    H, W = layout.shape
    assert np.all(layout[1, 1:-1] == CELL_PATH)
    assert np.all(layout[-2, 1:-1] == CELL_PATH)


def test_field_row_alternation():
    layout = generate_field_map(n_lanes=3, field_height=4)
    H, W = layout.shape
    # Field rows: rows 2 to H-3
    for row in range(2, H - 2):
        for col in range(1, W - 1):
            if col % 2 == 1:   # odd col → crop
                assert layout[row, col] == CELL_CROP, f"row={row}, col={col}"
            else:              # even col → path (lane)
                assert layout[row, col] == CELL_PATH, f"row={row}, col={col}"


def test_lane_adjacency():
    """Each field-row lane cell must have CROP on both sides."""
    layout = generate_field_map(n_lanes=3, field_height=4)
    H, W = layout.shape
    for row in range(2, H - 2):
        for col in range(2, W - 1, 2):  # even cols = lanes
            assert layout[row, col - 1] == CELL_CROP
            assert layout[row, col + 1] == CELL_CROP


def test_init_crop_states_shape():
    layout = generate_field_map(n_lanes=2, field_height=3)
    states = init_crop_states(layout, rng=np.random.default_rng(42))
    assert states.shape == layout.shape


def test_init_crop_states_only_on_crop_cells():
    layout = generate_field_map(n_lanes=2, field_height=3)
    states = init_crop_states(layout, rng=np.random.default_rng(42))
    H, W = layout.shape
    for r in range(H):
        for c in range(W):
            if layout[r, c] != CELL_CROP:
                assert states[r, c] == STATE_UNKNOWN, f"non-crop cell ({r},{c}) has state {states[r,c]}"
            else:
                assert states[r, c] in CROP_STATE_VALUES, f"crop cell ({r},{c}) has invalid state {states[r,c]}"


def test_init_crop_states_reproducible():
    layout = generate_field_map(n_lanes=2, field_height=3)
    s1 = init_crop_states(layout, rng=np.random.default_rng(0))
    s2 = init_crop_states(layout, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(s1, s2)
