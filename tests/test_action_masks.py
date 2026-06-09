# tests/test_action_masks.py
import numpy as np
import pytest
from env.farm_env import FarmEnv
from env.constants import (
    ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT,
    ACT_SCOUT, ACT_HARVEST, ACT_PEST,
    CELL_CROP, STATE_UNKNOWN, STATE_HARVEST_PENDING, STATE_PEST_PENDING,
)


def _env_at(pos, n_lanes=3, field_height=4, seed=0):
    env = FarmEnv(n_lanes=n_lanes, field_height=field_height)
    env.reset(seed=seed)
    env.agent_pos = pos
    return env


def test_scout_masked_on_headland():
    """헤드랜드에서는 인접 C 셀 없음 → Scout 마스킹."""
    env = _env_at((1, 2))  # top headland row
    masks = env.action_masks()
    assert not masks[ACT_SCOUT], "Scout should be masked on headland"


def test_scout_valid_in_lane():
    """필드 레인에서 미예찰 셀 있으면 Scout 활성화."""
    env = _env_at((2, 2))  # field row, lane col
    masks = env.action_masks()
    assert masks[ACT_SCOUT], "Scout should be valid in lane with unknown crops"


def test_scout_masked_after_all_adjacent_scouted():
    """인접 C 셀 전부 예찰 완료 시 Scout 마스킹."""
    env = _env_at((2, 2))
    adj = env._adjacent_crop_cells()
    for pos in adj:
        env.crop_states[pos] = 1  # STATE_NORMAL_DONE
    masks = env.action_masks()
    assert not masks[ACT_SCOUT], "Scout should be masked when all adjacent crops are done"


def test_harvest_masked_before_scout():
    """예찰 전에는 Harvest 마스킹."""
    env = _env_at((2, 2))
    masks = env.action_masks()
    assert not masks[ACT_HARVEST]


def test_harvest_valid_after_scout_reveals_harvest():
    """예찰 후 HARVEST_PENDING 셀이 있으면 Harvest 활성화."""
    env = _env_at((2, 2))
    adj = env._adjacent_crop_cells()
    env.crop_states[adj[0]] = STATE_HARVEST_PENDING
    masks = env.action_masks()
    assert masks[ACT_HARVEST]


def test_pest_valid_after_scout_reveals_pest():
    """예찰 후 PEST_PENDING 셀이 있으면 Pest 활성화."""
    env = _env_at((2, 2))
    adj = env._adjacent_crop_cells()
    env.crop_states[adj[0]] = STATE_PEST_PENDING
    masks = env.action_masks()
    assert masks[ACT_PEST]


def test_movement_masked_into_crop():
    """작물 셀 방향 이동 마스킹."""
    env = _env_at((2, 2))  # left=col1=CROP, right=col3=CROP
    masks = env.action_masks()
    assert not masks[ACT_LEFT]
    assert not masks[ACT_RIGHT]


def test_movement_valid_up_down_in_lane():
    """레인 안에서 상하 이동은 가능."""
    env = _env_at((3, 2))  # 중간 레인 (위아래 모두 PATH)
    masks = env.action_masks()
    assert masks[ACT_UP]
    assert masks[ACT_DOWN]
