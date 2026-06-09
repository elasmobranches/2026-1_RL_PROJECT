import numpy as np
import pytest
import gymnasium as gym
from env.farm_env import FarmEnv
from env.constants import CELL_PATH, N_ACTIONS


def test_env_registered_spaces():
    env = FarmEnv(n_lanes=3, field_height=4)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs), "reset obs out of obs_space bounds"
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == N_ACTIONS


def test_reset_obs_shape():
    env = FarmEnv(n_lanes=3, field_height=4)
    obs, info = env.reset(seed=42)
    H, W = env.H, env.W
    assert obs.shape == (4 * H * W,)


def test_reset_obs_range():
    env = FarmEnv(n_lanes=2, field_height=3)
    obs, _ = env.reset(seed=0)
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


def test_reset_agent_on_path():
    env = FarmEnv(n_lanes=3, field_height=4)
    env.reset(seed=0)
    r, c = env.agent_pos
    assert env.layout[r, c] == CELL_PATH


def test_reset_is_reproducible():
    env = FarmEnv(n_lanes=2, field_height=3)
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    np.testing.assert_array_equal(obs1, obs2)


def test_reset_clears_state():
    env = FarmEnv(n_lanes=2, field_height=3)
    env.reset(seed=0)
    # Take a step to dirty the state, then reset
    env.step(1)
    env.reset(seed=1)
    assert env.step_count == 0
    assert np.all(env.crop_states == 0)  # all unknown after reset
