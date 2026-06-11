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
    # obs = [completion(n_lanes), distances(n_lanes)]
    assert obs.shape == (env.n_lanes * 2,), f"Expected ({env.n_lanes * 2},), got {obs.shape}"


def test_obs_completion_all_zero_at_start():
    env = make_env()
    obs, _ = env.reset(seed=0)
    completion = obs[:env.n_lanes]
    assert np.all(completion == 0.0), f"Completion should be 0 at start, got {completion}"


def test_obs_distances_nonzero_at_start():
    """Agent starts at col 1 (lane 0), so distances to other lanes must be > 0."""
    env = make_env()
    obs, _ = env.reset(seed=0)
    distances = obs[env.n_lanes:]
    assert distances[0] == 0.0, "Distance to lane 0 (col 1) should be 0 at start"
    assert np.any(distances[1:] > 0.0), "Some lanes should be farther away"


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
    assert obs2.shape == (env.n_lanes * 2,)
    assert obs2.min() >= 0.0 and obs2.max() <= 1.0


def test_inner_target_lane_set_correctly():
    env = make_env()
    env.step(1)
    assert env.inner.target_lane_col == env.lane_cols[1]


def test_obs_increases_after_step():
    """Lane 0 coverage increases after processing it (robust: force normal crops)."""
    from env.constants import STATE_NORMAL_DONE
    env = HighLevelFarmEnv(_MockLowLevel(), n_beds=2, field_height=2)
    env.reset(seed=0)
    # Force lane 0 adjacent crops to be NORMAL so mock's scout action completes them
    lane0_col = env.lane_cols[0]
    for (r, c) in env.inner._adjacent_lane_crops(lane0_col):
        env.inner._true_states[r, c] = STATE_NORMAL_DONE
    obs_before = env._get_hl_obs().copy()
    env.step(0)
    obs_after = env._get_hl_obs()
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
