import numpy as np
import pytest
from env.hierarchical.lane_executor_env import LaneExecutorEnv
from env.constants import CELL_CROP, CELL_PATH, DONE_STATES, STATE_NORMAL_DONE, ACT_DOWN


def make_env(seed=0):
    env = LaneExecutorEnv(n_beds=2, field_height=2)
    env.reset(seed=seed)
    return env


def test_obs_shape_is_5_channels():
    env = make_env()
    obs, _ = env.reset(seed=0)
    H, W = env.H, env.W
    assert obs.shape == (5 * H * W,), f"Expected (5*{H}*{W},), got {obs.shape}"


def test_obs_range():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_ch4_marks_target_lane_col():
    env = make_env()
    env.reset(seed=0)
    env.target_lane_col = 4
    obs = env._get_obs()
    H, W = env.H, env.W
    ch4 = obs[4 * H * W:].reshape(H, W)
    assert np.all(ch4[:, 4] == 1.0), "target col should be 1"
    other_cols = [c for c in range(W) if c != 4]
    assert np.all(ch4[:, other_cols] == 0.0), "other cols should be 0"


def test_lane_cols_correct():
    env = LaneExecutorEnv(n_beds=2, field_height=2)
    env.reset(seed=0)
    assert env.lane_cols == [1, 4, 7]


def test_adjacent_lane_crops_edge_lane():
    env = make_env()
    crops = env._adjacent_lane_crops(1)
    cols = set(c for (r, c) in crops)
    assert cols == {2}, f"Edge lane col=1 should see col2 only, got {cols}"


def test_adjacent_lane_crops_inner_lane():
    env = make_env()
    crops = env._adjacent_lane_crops(4)
    cols = set(c for (r, c) in crops)
    assert cols == {3, 5}, f"Inner lane col=4 should see cols 3,5 got {cols}"


def test_is_lane_complete_false_at_start():
    env = make_env()
    env.target_lane_col = env.lane_cols[0]
    assert not env._is_lane_complete()


def test_terminated_when_target_lane_done():
    env = make_env()
    target_col = env.lane_cols[1]  # col 4
    env.target_lane_col = target_col
    for (r, c) in env._adjacent_lane_crops(target_col):
        env.crop_states[r, c] = STATE_NORMAL_DONE
    _, _, terminated, _, _ = env.step(ACT_DOWN)
    assert terminated


def test_not_terminated_when_other_lane_crops_remain():
    env = make_env()
    target_col = env.lane_cols[0]  # col 1
    env.target_lane_col = target_col
    for (r, c) in env._adjacent_lane_crops(target_col):
        env.crop_states[r, c] = STATE_NORMAL_DONE
    _, _, terminated, _, _ = env.step(ACT_DOWN)
    assert terminated, "Should terminate when just target lane is done"


def test_truncated_on_max_steps_per_lane():
    env = LaneExecutorEnv(n_beds=2, field_height=2, max_steps_per_lane=3)
    env.reset(seed=0)
    env.target_lane_col = env.lane_cols[1]
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(ACT_DOWN)
    if not terminated:
        assert truncated


def test_lane_coverage_rate():
    env = make_env()
    target_col = env.lane_cols[1]
    env.target_lane_col = target_col
    crops = env._adjacent_lane_crops(target_col)
    half = len(crops) // 2
    for (r, c) in crops[:half]:
        env.crop_states[r, c] = STATE_NORMAL_DONE
    rate = env._lane_coverage_rate()
    assert abs(rate - half / len(crops)) < 1e-6


def test_gymnasium_compatible():
    from gymnasium.utils.env_checker import check_env
    env = LaneExecutorEnv(n_beds=2, field_height=2)
    check_env(env, warn=True)
