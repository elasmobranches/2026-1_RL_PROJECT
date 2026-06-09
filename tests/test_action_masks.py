# tests/test_action_masks.py
import numpy as np
import pytest
from env.farm_env import FarmEnv
from env.constants import (
    ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT,
    ACT_SCOUT, ACT_HARVEST, ACT_PEST,
    CELL_CROP, CELL_PATH, STATE_UNKNOWN, STATE_HARVEST_PENDING, STATE_PEST_PENDING,
)

# New layout (n_beds=3, field_height=4):
#   W=12, H=8
#   Field row lane cols: 1, 4, 7, 10  ((col-1)%3==0)
#   Col 4 is an inner lane: left=col3=CROP, right=col5=CROP (both sides)
#   Col 1 is edge lane: left=col0=WALL, right=col2=CROP (one side only)


def _env_at(pos, n_beds=3, field_height=4, seed=0):
    env = FarmEnv(n_beds=n_beds, field_height=field_height)
    env.reset(seed=seed)
    env.agent_pos = pos
    return env


def test_scout_masked_on_headland():
    """Headland has no adjacent CROP → Scout masked."""
    env = _env_at((1, 4))  # top headland row, any col
    masks = env.action_masks()
    assert not masks[ACT_SCOUT], "Scout should be masked on headland"


def test_scout_valid_in_inner_lane():
    """Inner lane (col 4) has CROP on both sides → Scout valid."""
    env = _env_at((2, 4))  # field row, inner lane col
    masks = env.action_masks()
    assert masks[ACT_SCOUT], "Scout should be valid in lane with unknown crops"


def test_scout_valid_in_edge_lane():
    """Edge lane (col 1) has CROP on right only → Scout still valid."""
    env = _env_at((2, 1))  # field row, edge lane col
    masks = env.action_masks()
    assert masks[ACT_SCOUT], "Edge lane scout should be valid"


def test_scout_masked_after_all_adjacent_scouted():
    """All adjacent CROP cells done → Scout masked."""
    env = _env_at((2, 4))
    adj = env._adjacent_crop_cells()
    for pos in adj:
        env.crop_states[pos] = 1  # STATE_NORMAL_DONE
    masks = env.action_masks()
    assert not masks[ACT_SCOUT], "Scout should be masked when all adjacent crops are done"


def test_harvest_masked_before_scout():
    """Harvest masked when no HARVEST_PENDING adjacent."""
    env = _env_at((2, 4))
    masks = env.action_masks()
    assert not masks[ACT_HARVEST]


def test_harvest_valid_after_scout_reveals_harvest():
    """Harvest valid after adjacent crop revealed as HARVEST_PENDING."""
    env = _env_at((2, 4))
    adj = env._adjacent_crop_cells()
    env.crop_states[adj[0]] = STATE_HARVEST_PENDING
    masks = env.action_masks()
    assert masks[ACT_HARVEST]


def test_pest_valid_after_scout_reveals_pest():
    """Pest valid after adjacent crop revealed as PEST_PENDING."""
    env = _env_at((2, 4))
    adj = env._adjacent_crop_cells()
    env.crop_states[adj[0]] = STATE_PEST_PENDING
    masks = env.action_masks()
    assert masks[ACT_PEST]


def test_movement_masked_into_crop():
    """Left/right movement from inner lane blocked by CROP."""
    env = _env_at((2, 4))  # col4: left=col3=CROP, right=col5=CROP
    masks = env.action_masks()
    assert not masks[ACT_LEFT]
    assert not masks[ACT_RIGHT]


def test_movement_valid_up_down_in_lane():
    """Up/down movement valid from mid-field lane."""
    env = _env_at((3, 4))  # middle field row, inner lane
    masks = env.action_masks()
    assert masks[ACT_UP]
    assert masks[ACT_DOWN]


def test_outer_crop_not_adjacent_to_inner_lane():
    """Outer crop col is NOT adjacent to inner lane — only scoutable from other lane."""
    env = _env_at((2, 4))  # inner lane col=4; left=col3(inner crop), right=col5(inner crop)
    adj_cols = [c for (r, c) in env._adjacent_crop_cells()]
    # col 2 (outer crop of first bed) should NOT be in adj from col 4
    assert 2 not in adj_cols, "Outer crop col 2 must not be visible from lane col 4"
    # col 6 (outer crop of second bed) should NOT be in adj from col 4
    assert 6 not in adj_cols, "Outer crop col 6 must not be visible from lane col 4"
