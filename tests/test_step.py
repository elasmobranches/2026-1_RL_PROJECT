# tests/test_step.py
# New layout: n_beds=2 → W=9, H=6, lane cols in field rows: 1, 4, 7
# Col 4 = inner lane (CROP left=col3, right=col5)
# Col 1 = edge lane (WALL left=col0, CROP right=col2)
import numpy as np
import pytest
from env.farm_env import FarmEnv
from env.constants import (
    ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT,
    ACT_SCOUT, ACT_HARVEST, ACT_PEST,
    REWARD_STEP, REWARD_COLLISION, REWARD_SCOUT_NEW, REWARD_NORMAL_CONFIRM,
    REWARD_HARVEST, REWARD_PEST, REWARD_COMPLETION,
    STATE_UNKNOWN, STATE_NORMAL_DONE, STATE_HARVEST_PENDING, STATE_PEST_PENDING,
    STATE_HARVEST_DONE, STATE_PEST_DONE, DONE_STATES,
    CELL_PATH,
)


def make_env(seed=0):
    env = FarmEnv(n_beds=2, field_height=2, max_steps=500)
    env.reset(seed=seed)
    return env


# --- Movement ---

def test_move_down_from_headland():
    env = make_env()
    start = env.agent_pos          # (1, 1) — top headland, first lane
    obs, reward, term, trunc, info = env.step(ACT_DOWN)
    assert env.agent_pos == (start[0] + 1, start[1])
    assert abs(reward - REWARD_STEP) < 1e-6


def test_collision_into_crop():
    env = make_env()
    env.agent_pos = (2, 4)         # inner lane col 4 in field row
    obs, reward, term, trunc, info = env.step(ACT_LEFT)  # col3 = CROP
    assert env.agent_pos == (2, 4)  # position unchanged
    assert reward < REWARD_STEP     # collision penalty applied


def test_step_count_increments():
    env = make_env()
    for i in range(3):
        env.step(ACT_DOWN)
    assert env.step_count == 3


# --- Scout ---

def test_scout_reveals_adjacent_cells():
    env = make_env()
    env.agent_pos = (2, 4)         # inner lane: left=col3, right=col5
    env._true_states[2, 3] = STATE_NORMAL_DONE
    env._true_states[2, 5] = STATE_HARVEST_PENDING
    env.crop_states[:] = STATE_UNKNOWN

    env.step(ACT_SCOUT)
    assert env.crop_states[2, 3] == STATE_NORMAL_DONE
    assert env.crop_states[2, 5] == STATE_HARVEST_PENDING


def test_scout_reward_normal():
    env = make_env()
    env.agent_pos = (2, 4)
    adj = env._adjacent_crop_cells()
    for pos in adj:
        env._true_states[pos] = STATE_NORMAL_DONE
    env.crop_states[:] = STATE_UNKNOWN

    _, reward, _, _, _ = env.step(ACT_SCOUT)
    expected = REWARD_STEP + len(adj) * (REWARD_SCOUT_NEW + REWARD_NORMAL_CONFIRM)
    assert abs(reward - expected) < 1e-5


def test_scout_reward_harvest_pending():
    env = make_env()
    env.agent_pos = (2, 4)
    adj = env._adjacent_crop_cells()
    for pos in adj:
        env._true_states[pos] = STATE_HARVEST_PENDING
    env.crop_states[:] = STATE_UNKNOWN

    _, reward, _, _, _ = env.step(ACT_SCOUT)
    expected = REWARD_STEP + len(adj) * REWARD_SCOUT_NEW
    assert abs(reward - expected) < 1e-5


def test_edge_lane_scouts_one_crop_only():
    """Edge lane (col 1) only has CROP on right — scout sees exactly 1 cell."""
    env = make_env()
    env.agent_pos = (2, 1)         # edge lane: left=WALL, right=col2=CROP
    adj = env._adjacent_crop_cells()
    assert len(adj) == 1
    assert adj[0] == (2, 2)


# --- Harvest ---

def test_harvest_action():
    env = make_env()
    env.agent_pos = (2, 4)
    adj = env._adjacent_crop_cells()
    for pos in adj:
        env.crop_states[pos] = STATE_HARVEST_PENDING

    _, reward, _, _, _ = env.step(ACT_HARVEST)
    for pos in adj:
        assert env.crop_states[pos] == STATE_HARVEST_DONE
    assert abs(reward - (REWARD_STEP + len(adj) * REWARD_HARVEST)) < 1e-5


# --- Pest ---

def test_pest_action():
    env = make_env()
    env.agent_pos = (2, 4)
    adj = env._adjacent_crop_cells()
    for pos in adj:
        env.crop_states[pos] = STATE_PEST_PENDING

    _, reward, _, _, _ = env.step(ACT_PEST)
    for pos in adj:
        assert env.crop_states[pos] == STATE_PEST_DONE
    assert abs(reward - (REWARD_STEP + len(adj) * REWARD_PEST)) < 1e-5


# --- Termination ---

def test_terminated_when_all_done():
    env = make_env()
    for pos in env._crop_cells:
        env.crop_states[pos] = STATE_NORMAL_DONE
    last = env._crop_cells[-1]
    r_last, c_last = last
    env.crop_states[last] = STATE_HARVEST_PENDING
    # Find adjacent PATH cell
    if c_last + 1 < env.W and env.layout[r_last, c_last + 1] == CELL_PATH:
        env.agent_pos = (r_last, c_last + 1)
    else:
        env.agent_pos = (r_last, c_last - 1)

    _, reward, terminated, truncated, _ = env.step(ACT_HARVEST)
    assert terminated
    assert not truncated
    assert reward >= REWARD_COMPLETION


def test_truncated_on_max_steps():
    env = FarmEnv(n_beds=2, field_height=2, max_steps=3)
    env.reset(seed=0)
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(ACT_DOWN)
    if not terminated:
        assert truncated


def test_obs_valid_after_step():
    env = make_env()
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)
