import numpy as np
import pytest
from env.hierarchical.high_level_env import HighLevelFarmEnv
from env.constants import STATE_NORMAL_DONE


class _MockLowLevel:
    """Mock Low-level: Scout if possible (action 4), else move DOWN (action 1)."""
    def predict(self, obs, deterministic=True, action_masks=None):
        if action_masks is not None and action_masks[4]:
            return np.array(4), None   # ACT_SCOUT
        return np.array(1), None       # ACT_DOWN


def make_env(seed=0):
    env = HighLevelFarmEnv(_MockLowLevel(), n_beds=2, field_height=2)
    env.reset(seed=seed)
    return env


def test_obs_shape():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.n_lanes,), f"Expected ({env.n_lanes},), got {obs.shape}"


def test_obs_all_zero_at_start():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert np.all(obs == 0.0), f"Expected all zeros, got {obs}"


def test_action_space():
    env = make_env()
    assert env.action_space.n == env.n_lanes


def test_n_lanes_correct():
    env = make_env()
    assert env.n_lanes == 3
    assert env.lane_cols == [1, 4, 7]


def test_step_returns_valid_obs():
    env = make_env()
    obs, _ = env.reset(seed=0)
    obs2, reward, terminated, truncated, info = env.step(0)
    assert obs2.shape == (env.n_lanes,)
    assert obs2.min() >= 0.0 and obs2.max() <= 1.0


def test_inner_target_lane_set_correctly():
    env = make_env()
    env.step(1)
    assert env.inner.target_lane_col == env.lane_cols[1]


def test_obs_increases_after_step():
    env = make_env()
    env.reset(seed=0)
    obs_before, _ = env.reset(seed=0)
    obs_after, _, _, _, _ = env.step(0)
    assert obs_after[0] > obs_before[0], "Lane 0 coverage should increase after step"


def test_terminates_when_all_done():
    env = make_env()
    env.reset(seed=0)
    for pos in env.inner._crop_cells:
        env.inner.crop_states[pos] = STATE_NORMAL_DONE
    _, _, terminated, _, _ = env.step(0)
    assert terminated


def test_truncated_on_max_visits():
    env = HighLevelFarmEnv(_MockLowLevel(), n_beds=2, field_height=2, max_lane_visits=2)
    env.reset(seed=0)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(0)
    if not terminated:
        assert truncated


def test_info_contains_steps_for_lane():
    env = make_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    assert "steps_for_lane" in info
    assert info["steps_for_lane"] > 0


def test_inner_state_persists_across_steps():
    env = make_env()
    env.reset(seed=0)
    env.step(0)  # process lane 0
    snapshot = env.inner.crop_states.copy()
    env.step(1)  # process lane 1
    for (r, c) in env.inner._adjacent_lane_crops(env.lane_cols[0]):
        assert env.inner.crop_states[r, c] == snapshot[r, c], \
            "Previously processed crops must not be reset"
