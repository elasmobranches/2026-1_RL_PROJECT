import numpy as np
import pytest
from env.constants import CELL_PATH, CELL_CROP, CELL_WALL, STATE_UNKNOWN, DONE_STATES, CROP_STATE_VALUES
from env.map_generator import generate_field_map, init_crop_states


def test_map_shape():
    layout = generate_field_map(n_beds=3, field_height=4)
    expected_H = 4 + 4    # field_height + 2 walls + 2 headlands
    expected_W = 3*3 + 3  # 3*n_beds + 3
    assert layout.shape == (expected_H, expected_W)


def test_walls_on_border():
    layout = generate_field_map(n_beds=3, field_height=4)
    H, W = layout.shape
    assert np.all(layout[0, :] == CELL_WALL)
    assert np.all(layout[-1, :] == CELL_WALL)
    assert np.all(layout[:, 0] == CELL_WALL)
    assert np.all(layout[:, -1] == CELL_WALL)


def test_headland_rows_are_path():
    layout = generate_field_map(n_beds=3, field_height=4)
    H, W = layout.shape
    assert np.all(layout[1, 1:-1] == CELL_PATH)
    assert np.all(layout[-2, 1:-1] == CELL_PATH)


def test_field_row_pattern():
    """Field rows follow P CC P CC P pattern: (col-1)%3==0 → PATH, else CROP."""
    layout = generate_field_map(n_beds=3, field_height=4)
    H, W = layout.shape
    for row in range(2, H - 2):
        for col in range(1, W - 1):
            if (col - 1) % 3 == 0:
                assert layout[row, col] == CELL_PATH, f"row={row}, col={col} should be PATH"
            else:
                assert layout[row, col] == CELL_CROP, f"row={row}, col={col} should be CROP"


def test_lane_scouts_only_adjacent_crop():
    """Each driving lane (PATH) sees at most 1 CROP on each side (not 2)."""
    layout = generate_field_map(n_beds=3, field_height=4)
    H, W = layout.shape
    for row in range(2, H - 2):
        for col in range(1, W - 1):
            if layout[row, col] == CELL_PATH:
                left  = layout[row, col-1] if col-1 >= 0 else CELL_WALL
                right = layout[row, col+1] if col+1 < W  else CELL_WALL
                # Each adjacent cell is either CROP or WALL — never another PATH
                assert left  in (CELL_CROP, CELL_WALL), f"col={col}: left is PATH?"
                assert right in (CELL_CROP, CELL_WALL), f"col={col}: right is PATH?"


def test_two_col_bed_outer_not_adjacent_to_same_lane():
    """The OUTER crop column of a bed is NOT adjacent to the same lane as the inner col."""
    layout = generate_field_map(n_beds=2, field_height=4)
    # Layout inner cols: P C C P C C P  (cols 1..6 for n_beds=2, W=9)
    # col 1=P, col 2=C(inner), col 3=C(outer), col 4=P, col 5=C(inner), col 6=C(outer), col 7=P
    H, W = layout.shape
    for row in range(2, H - 2):
        # outer crop cols: (col-1)%3 == 2
        for col in range(1, W - 1):
            if (col - 1) % 3 == 2:  # outer crop of a bed
                left_lane  = col - 2  # 2 steps away
                right_lane = col + 1  # immediately right
                # outer col is NOT adjacent (distance 1) to the lane on its outer side
                assert layout[row, col-1] == CELL_CROP, \
                    f"outer crop at col={col}: left neighbour should be inner crop, not lane"


def test_init_crop_states_shape():
    layout = generate_field_map(n_beds=2, field_height=3)
    states = init_crop_states(layout, rng=np.random.default_rng(42))
    assert states.shape == layout.shape


def test_init_crop_states_only_on_crop_cells():
    layout = generate_field_map(n_beds=2, field_height=3)
    states = init_crop_states(layout, rng=np.random.default_rng(42))
    H, W = layout.shape
    for r in range(H):
        for c in range(W):
            if layout[r, c] != CELL_CROP:
                assert states[r, c] == STATE_UNKNOWN, f"non-crop cell ({r},{c}) has state {states[r,c]}"
            else:
                assert states[r, c] in CROP_STATE_VALUES, f"crop cell ({r},{c}) has invalid state {states[r,c]}"


def test_init_crop_states_reproducible():
    layout = generate_field_map(n_beds=2, field_height=3)
    s1 = init_crop_states(layout, rng=np.random.default_rng(0))
    s2 = init_crop_states(layout, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(s1, s2)


def test_init_crop_states_probability_distribution():
    """Crop states should roughly follow CROP_STATE_PROBS distribution."""
    from env.constants import CROP_STATE_PROBS, CROP_STATE_VALUES
    layout = generate_field_map(n_beds=10, field_height=50)
    states = init_crop_states(layout, rng=np.random.default_rng(0))
    crop_cells = states[layout == 1]
    total = len(crop_cells)
    for state_val, expected_prob in zip(CROP_STATE_VALUES, CROP_STATE_PROBS):
        actual = np.sum(crop_cells == state_val) / total
        assert abs(actual - expected_prob) < 0.05, (
            f"State {state_val}: expected ~{expected_prob:.0%}, got {actual:.1%}"
        )


def test_generate_field_map_invalid_inputs():
    with pytest.raises(ValueError):
        generate_field_map(n_beds=0, field_height=4)
    with pytest.raises(ValueError):
        generate_field_map(n_beds=3, field_height=0)
