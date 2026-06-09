# Hierarchical RL (Step 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 계층적 RL로 경로 효율성 개선 — High-level 정책이 레인 방문 순서를 결정하고, Low-level 정책이 해당 레인을 처리하여 평균 완료 스텝을 146 → 110 이하로 줄인다.

**Architecture:** LaneExecutorEnv(Low-level, MaskablePPO)가 지정된 레인만 처리하고, HighLevelFarmEnv(High-level, PPO)가 어느 레인을 다음에 방문할지 선택한다. Low-level은 FarmEnv를 상속하여 ch4(목표 레인 인디케이터)를 추가한 5채널 obs를 사용한다.

**Tech Stack:** gymnasium, sb3-contrib MaskablePPO (low-level), stable-baselines3 PPO (high-level), pytest

---

## 파일 구조

```
rlproject/
├── env/
│   ├── hierarchical/
│   │   ├── __init__.py           # 패키지
│   │   ├── lane_executor_env.py  # Low-level env (LaneExecutorEnv)
│   │   └── high_level_env.py     # High-level env (HighLevelFarmEnv)
│   └── ... (기존 파일 유지)
├── tests/
│   ├── test_lane_executor.py     # LaneExecutorEnv 테스트
│   └── test_high_level_env.py    # HighLevelFarmEnv 테스트
├── train_hierarchical.py         # Phase 1 + Phase 2 학습
└── evaluate_hierarchical.py      # Step 1 vs Step 2 비교 분석
```

---

## 환경 구조 참고 (n_beds=4, field_height=8)

```
H=12, W=15
Lane cols: [1, 4, 7, 10, 13]  (5개 레인)
Field rows: 2..9               (8행)

Layout: W P CC P CC P CC P CC P W
              ↑ 각 CC = 2열 재배단
              레인 col=4 → 인접 작물: col3(8셀) + col5(8셀) = 16셀
              레인 col=1 → 인접 작물: col2(8셀)만 = 8셀 (엣지)
```

High-level obs: `[lane0_done_rate, lane1_done_rate, ..., lane4_done_rate]` (5-dim)
High-level action: `0~4` (어느 레인 인덱스를 다음에 방문할지)

---

## Task 1: 패키지 셋업 + REWARD_LANE_COMPLETE 상수 추가

**Files:**
- Create: `env/hierarchical/__init__.py`
- Modify: `env/constants.py`

- [ ] **Step 1: env/constants.py 맨 아래에 상수 추가**

```python
# Hierarchical RL (Step 2)
REWARD_LANE_COMPLETE = 10.0   # Low-level: 레인 완료 보상
REWARD_HL_LANE_DONE  =  5.0   # High-level: 레인 완료 보상
REWARD_HL_ALL_DONE   = 20.0   # High-level: 전체 완료 보상
HL_STEP_COST         =  0.01  # High-level: 스텝당 비용 계수
```

- [ ] **Step 2: `env/hierarchical/__init__.py` 작성**

```python
from env.hierarchical.lane_executor_env import LaneExecutorEnv
from env.hierarchical.high_level_env import HighLevelFarmEnv
```

- [ ] **Step 3: 임포트 동작 확인**

```bash
python -c "from env.constants import REWARD_LANE_COMPLETE; print('OK', REWARD_LANE_COMPLETE)"
```

Expected: `OK 10.0`

- [ ] **Step 4: Commit**

```bash
git add env/constants.py env/hierarchical/__init__.py
git commit -m "feat: add hierarchical RL package and reward constants"
```

---

## Task 2: LaneExecutorEnv + 테스트 (TDD)

**Files:**
- Create: `env/hierarchical/lane_executor_env.py`
- Create: `tests/test_lane_executor.py`

- [ ] **Step 1: 테스트 먼저 작성**

```python
# tests/test_lane_executor.py
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
    """ch4(5번째 채널)의 target_lane_col 열이 1.0, 나머지 0.0."""
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
    """n_beds=2 → lane cols [1, 4, 7]."""
    env = LaneExecutorEnv(n_beds=2, field_height=2)
    env.reset(seed=0)
    assert env.lane_cols == [1, 4, 7]


def test_adjacent_lane_crops_edge_lane():
    """엣지 레인(col=1)은 오른쪽 한 열만 인접."""
    env = make_env()
    crops = env._adjacent_lane_crops(1)
    cols = set(c for (r, c) in crops)
    assert cols == {2}, f"Edge lane col=1 should see col2 only, got {cols}"


def test_adjacent_lane_crops_inner_lane():
    """내부 레인(col=4)은 col3, col5 양쪽 인접."""
    env = make_env()
    crops = env._adjacent_lane_crops(4)
    cols = set(c for (r, c) in crops)
    assert cols == {3, 5}, f"Inner lane col=4 should see cols 3,5 got {cols}"


def test_is_lane_complete_false_at_start():
    env = make_env()
    env.target_lane_col = env.lane_cols[0]
    assert not env._is_lane_complete()


def test_terminated_when_target_lane_done():
    """목표 레인 인접 작물 전부 DONE → terminated."""
    env = make_env()
    target_col = env.lane_cols[1]  # col 4
    env.target_lane_col = target_col
    for (r, c) in env._adjacent_lane_crops(target_col):
        env.crop_states[r, c] = STATE_NORMAL_DONE
    _, _, terminated, _, _ = env.step(ACT_DOWN)
    assert terminated


def test_not_terminated_when_other_lane_crops_remain():
    """다른 레인 작물이 남아도 목표 레인만 완료되면 terminated."""
    env = make_env()
    target_col = env.lane_cols[0]  # col 1
    env.target_lane_col = target_col
    # 목표 레인만 완료
    for (r, c) in env._adjacent_lane_crops(target_col):
        env.crop_states[r, c] = STATE_NORMAL_DONE
    # 다른 레인 작물은 그대로
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
    # Mark half done
    half = len(crops) // 2
    for (r, c) in crops[:half]:
        env.crop_states[r, c] = STATE_NORMAL_DONE
    rate = env._lane_coverage_rate()
    assert abs(rate - half / len(crops)) < 1e-6


def test_gymnasium_compatible():
    from gymnasium.utils.env_checker import check_env
    env = LaneExecutorEnv(n_beds=2, field_height=2)
    check_env(env, warn=True)
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/test_lane_executor.py -v 2>/dev/null | tail -5
```

Expected: `ModuleNotFoundError: No module named 'env.hierarchical.lane_executor_env'`

- [ ] **Step 3: LaneExecutorEnv 구현**

```python
# env/hierarchical/lane_executor_env.py
from __future__ import annotations
import numpy as np
from gymnasium import spaces
from env.farm_env import FarmEnv
from env.constants import (
    CELL_CROP, DONE_STATES,
    N_ACTIONS, MOVE_DELTA,
    ACT_SCOUT, ACT_HARVEST, ACT_PEST,
    REWARD_STEP, REWARD_COLLISION,
    REWARD_SCOUT_NEW, REWARD_NORMAL_CONFIRM,
    REWARD_HARVEST, REWARD_PEST,
    REWARD_LANE_COMPLETE,
)


class LaneExecutorEnv(FarmEnv):
    """
    Low-level env: navigate to target lane and process all adjacent crops.

    Extends FarmEnv with:
    - ch4 (target lane indicator) added to observation → 5*H*W obs
    - Terminates when all crops adjacent to target_lane_col are in DONE_STATES
    - max_steps_per_lane replaces max_steps for episode limit
    """

    def __init__(
        self,
        n_beds: int = 4,
        field_height: int = 8,
        max_steps_per_lane: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__(n_beds=n_beds, field_height=field_height, render_mode=render_mode)

        self.lane_cols: list[int] = [c for c in range(1, self.W - 1) if (c - 1) % 3 == 0]
        self.max_steps_per_lane: int = max_steps_per_lane or self.H * self.W
        self.target_lane_col: int = self.lane_cols[0]

        # Override observation space: 5 channels
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5 * self.H * self.W,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed)
        self.target_lane_col = (options or {}).get("target_lane_col", self.lane_cols[0])
        self.step_count = 0
        return self._get_obs(), info

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()          # 4 * H * W
        ch4 = np.zeros((self.H, self.W), dtype=np.float32)
        ch4[:, self.target_lane_col] = 1.0
        return np.concatenate([base, ch4.ravel()])

    # ------------------------------------------------------------------
    def step(self, action: int):
        self.step_count += 1
        reward = REWARD_STEP

        if action in MOVE_DELTA:
            reward += self._handle_move(action)
        elif action == ACT_SCOUT:
            reward += self._handle_scout()
        elif action == ACT_HARVEST:
            reward += self._handle_harvest()
        elif action == ACT_PEST:
            reward += self._handle_pest()

        lane_done = self._is_lane_complete()
        if lane_done:
            reward += REWARD_LANE_COMPLETE

        terminated = lane_done
        truncated = (self.step_count >= self.max_steps_per_lane) and not terminated

        if self.render_mode == "human":
            self.render()

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            {
                "coverage": self._coverage_rate(),
                "lane_coverage": self._lane_coverage_rate(),
                "step": self.step_count,
            },
        )

    # ------------------------------------------------------------------
    def _is_lane_complete(self) -> bool:
        """All crops adjacent to target_lane_col are in DONE_STATES."""
        return all(
            self.crop_states[r, c] in DONE_STATES
            for (r, c) in self._adjacent_lane_crops(self.target_lane_col)
        )

    def _adjacent_lane_crops(self, lane_col: int) -> list[tuple[int, int]]:
        """All CELL_CROP cells directly adjacent to lane_col (field rows only)."""
        result = []
        for row in range(2, self.H - 2):
            for dc in (-1, 1):
                nc = lane_col + dc
                if 0 <= nc < self.W and self.layout[row, nc] == CELL_CROP:
                    result.append((row, nc))
        return result

    def _lane_coverage_rate(self) -> float:
        crops = self._adjacent_lane_crops(self.target_lane_col)
        if not crops:
            return 1.0
        done = sum(1 for (r, c) in crops if self.crop_states[r, c] in DONE_STATES)
        return done / len(crops)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_lane_executor.py -v 2>/dev/null | tail -5
```

Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add env/hierarchical/lane_executor_env.py tests/test_lane_executor.py
git commit -m "feat: LaneExecutorEnv with 5-channel obs and lane-targeted termination"
```

---

## Task 3: HighLevelFarmEnv + 테스트 (TDD)

**Files:**
- Create: `env/hierarchical/high_level_env.py`
- Create: `tests/test_high_level_env.py`

- [ ] **Step 1: 테스트 먼저 작성**

```python
# tests/test_high_level_env.py
import numpy as np
import pytest
from env.hierarchical.high_level_env import HighLevelFarmEnv
from env.constants import STATE_NORMAL_DONE


class _MockLowLevel:
    """Mock Low-level: Scout if possible, else move DOWN."""
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
    """에피소드 시작 시 모든 레인 완료율 0."""
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert np.all(obs == 0.0), f"Expected all zeros, got {obs}"


def test_action_space():
    env = make_env()
    assert env.action_space.n == env.n_lanes


def test_n_lanes_correct():
    """n_beds=2 → 3개 레인 (cols 1, 4, 7)."""
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
    """step(action=1) → inner.target_lane_col = lane_cols[1]."""
    env = make_env()
    env.step(1)
    assert env.inner.target_lane_col == env.lane_cols[1]


def test_obs_increases_after_step():
    """레인 처리 후 해당 레인의 완료율이 올라가야 함."""
    env = make_env()
    env.reset(seed=0)
    obs_before, _ = env.reset(seed=0)
    obs_after, _, _, _, _ = env.step(0)
    # 레인 0이 적어도 어느 정도 처리됨
    assert obs_after[0] > obs_before[0], "Lane 0 coverage should increase after step"


def test_terminates_when_all_done():
    """모든 작물 처리 시 terminated."""
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
    """step() 호출 사이에 inner env crop_states가 유지됨."""
    env = make_env()
    env.reset(seed=0)
    env.step(0)  # process lane 0
    snapshot = env.inner.crop_states.copy()
    env.step(1)  # process lane 1
    # lane 0 crops should remain done
    for (r, c) in env.inner._adjacent_lane_crops(env.lane_cols[0]):
        assert env.inner.crop_states[r, c] == snapshot[r, c], \
            "Previously processed crops must not be reset"
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/test_high_level_env.py -v 2>/dev/null | tail -5
```

Expected: `ModuleNotFoundError: No module named 'env.hierarchical.high_level_env'`

- [ ] **Step 3: HighLevelFarmEnv 구현**

```python
# env/hierarchical/high_level_env.py
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.hierarchical.lane_executor_env import LaneExecutorEnv
from env.constants import DONE_STATES, REWARD_HL_LANE_DONE, REWARD_HL_ALL_DONE, HL_STEP_COST


class HighLevelFarmEnv(gym.Env):
    """
    High-level env: select which lane to visit next.

    Each step() executes the low-level policy until target lane is complete
    (or max_steps_per_lane exceeded). The inner LaneExecutorEnv state persists
    across high-level steps — crops processed in earlier lanes stay processed.

    Observation: fraction of adjacent crops done per lane (n_lanes-dim float32)
    Action:      lane index to visit next (Discrete(n_lanes))
    """

    def __init__(
        self,
        low_level_model,
        n_beds: int = 4,
        field_height: int = 8,
        max_lane_visits: int | None = None,
    ):
        super().__init__()
        self.low_level_model = low_level_model
        self.inner = LaneExecutorEnv(n_beds=n_beds, field_height=field_height)
        self.lane_cols: list[int] = self.inner.lane_cols
        self.n_lanes: int = len(self.lane_cols)
        self.max_lane_visits: int = max_lane_visits or self.n_lanes * 3

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_lanes,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_lanes)
        self._lane_visits: int = 0

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        self.inner.reset(seed=seed)
        self._lane_visits = 0
        return self._get_hl_obs(), {}

    # ------------------------------------------------------------------
    def _get_hl_obs(self) -> np.ndarray:
        obs = np.zeros(self.n_lanes, dtype=np.float32)
        for i, lane_col in enumerate(self.lane_cols):
            crops = self.inner._adjacent_lane_crops(lane_col)
            if crops:
                done = sum(1 for (r, c) in crops if self.inner.crop_states[r, c] in DONE_STATES)
                obs[i] = done / len(crops)
        return obs

    # ------------------------------------------------------------------
    def step(self, action: int):
        target_col = self.lane_cols[action]
        self.inner.target_lane_col = target_col
        self.inner.step_count = 0   # reset per-lane step counter

        ll_obs = self.inner._get_obs()
        lane_done = lane_trunc = False
        total_steps = 0
        last_info: dict = {}

        while not (lane_done or lane_trunc):
            ll_action, _ = self.low_level_model.predict(
                ll_obs,
                deterministic=True,
                action_masks=self.inner.action_masks(),
            )
            ll_obs, _, lane_done, lane_trunc, last_info = self.inner.step(int(ll_action))
            total_steps += 1

        self._lane_visits += 1

        hl_reward = -total_steps * HL_STEP_COST
        if lane_done:
            hl_reward += REWARD_HL_LANE_DONE

        all_done = self.inner._is_complete()
        if all_done:
            hl_reward += REWARD_HL_ALL_DONE

        terminated = all_done
        truncated = (self._lane_visits >= self.max_lane_visits) and not terminated

        return (
            self._get_hl_obs(),
            float(hl_reward),
            terminated,
            truncated,
            {
                "steps_for_lane": total_steps,
                "coverage": last_info.get("coverage", 0.0),
                "lane_visits": self._lane_visits,
            },
        )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_high_level_env.py -v 2>/dev/null | tail -5
```

Expected: `11 passed`

- [ ] **Step 5: 전체 테스트 스위트 확인**

```bash
pytest tests/ -v 2>/dev/null | tail -5
```

Expected: `63 passed` (기존 40 + 신규 23)

- [ ] **Step 6: Commit**

```bash
git add env/hierarchical/high_level_env.py tests/test_high_level_env.py
git commit -m "feat: HighLevelFarmEnv with lane-sequencing high-level policy"
```

---

## Task 4: Phase 1 학습 — LaneExecutorEnv (Low-level)

**Files:**
- Create: `train_hierarchical.py`

- [ ] **Step 1: train_hierarchical.py 작성**

```python
# train_hierarchical.py
import os
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.hierarchical.lane_executor_env import LaneExecutorEnv
from env.hierarchical.high_level_env import HighLevelFarmEnv


def mask_fn(env):
    return env.action_masks()


# ── Phase 1: Low-level (LaneExecutorEnv) ──────────────────────────────

class RandomLaneWrapper(gym.Wrapper):
    """에피소드마다 무작위 레인을 목표로 설정."""
    def __init__(self, env):
        super().__init__(env)
        self._rng = np.random.default_rng()

    def reset(self, **kwargs):
        if kwargs.get("seed") is not None:
            self._rng = np.random.default_rng(kwargs["seed"])
        lane_idx = int(self._rng.integers(len(self.unwrapped.lane_cols)))
        target_col = self.unwrapped.lane_cols[lane_idx]
        kwargs.setdefault("options", {})["target_lane_col"] = target_col
        return self.env.reset(**kwargs)


def make_lane_env(seed=0):
    def _init():
        env = LaneExecutorEnv(n_beds=4, field_height=8)
        env = RandomLaneWrapper(env)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init


def train_low_level(total_timesteps=500_000, save_path="models/lane_executor"):
    os.makedirs("models", exist_ok=True)
    vec_env = DummyVecEnv([make_lane_env(seed=i) for i in range(4)])

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        seed=0,
    )
    print("=== Phase 1: Training Low-level LaneExecutorEnv ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.close()
    print(f"Low-level model saved to {save_path}.zip")
    return model


# ── Phase 2: High-level (HighLevelFarmEnv) ────────────────────────────

def make_hl_env(low_level_model, seed=0):
    def _init():
        env = HighLevelFarmEnv(low_level_model, n_beds=4, field_height=8)
        env = Monitor(env)
        return env
    return _init


def train_high_level(low_level_model, total_timesteps=200_000, save_path="models/high_level"):
    vec_env = DummyVecEnv([make_hl_env(low_level_model, seed=i) for i in range(4)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        seed=0,
    )
    print("=== Phase 2: Training High-level HighLevelFarmEnv ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.close()
    print(f"High-level model saved to {save_path}.zip")
    return model


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ll_model = train_low_level(total_timesteps=500_000)
    hl_model = train_high_level(ll_model, total_timesteps=200_000)
    print("=== Hierarchical training complete ===")
```

- [ ] **Step 2: 스모크 테스트 (100 스텝으로 동작 확인)**

```bash
python -c "
import os; os.makedirs('models', exist_ok=True)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env.hierarchical.lane_executor_env import LaneExecutorEnv
import gymnasium as gym

class _RLW(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._rng = np.random.default_rng()
    def reset(self, **kw):
        idx = int(self._rng.integers(len(self.unwrapped.lane_cols)))
        kw.setdefault('options', {})['target_lane_col'] = self.unwrapped.lane_cols[idx]
        return self.env.reset(**kw)

def make():
    env = LaneExecutorEnv(n_beds=2, field_height=2)
    env = _RLW(env)
    env = ActionMasker(env, lambda e: e.action_masks())
    return Monitor(env)

vec_env = DummyVecEnv([make])
model = MaskablePPO('MlpPolicy', vec_env, verbose=0, seed=0)
model.learn(total_timesteps=100)
print('Phase 1 smoke test: OK')
vec_env.close()
" 2>/dev/null
```

Expected: `Phase 1 smoke test: OK`

- [ ] **Step 3: 본격 학습 실행**

```bash
python train_hierarchical.py 2>/dev/null | grep -E "ep_rew_mean|Phase|saved"
```

Expected: Phase 1 → ep_rew_mean 상승, Phase 2 → ep_rew_mean 상승, 두 모델 저장.

- [ ] **Step 4: 모델 파일 존재 확인**

```bash
ls -lh models/lane_executor.zip models/high_level.zip
```

Expected: 두 파일 모두 존재.

- [ ] **Step 5: Commit**

```bash
git add train_hierarchical.py
git commit -m "feat: hierarchical training script (Phase 1 LaneExecutor + Phase 2 HighLevel)"
```

---

## Task 5: 평가 비교 스크립트 + 보고서 결과 업데이트

**Files:**
- Create: `evaluate_hierarchical.py`

- [ ] **Step 1: evaluate_hierarchical.py 작성**

```python
# evaluate_hierarchical.py
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from env.farm_env import FarmEnv
from env.hierarchical.lane_executor_env import LaneExecutorEnv
from env.hierarchical.high_level_env import HighLevelFarmEnv


# ── Step 1 baseline ───────────────────────────────────────────────────

def run_flat_episode(model, env, seed):
    obs, _ = env.reset(seed=seed)
    total_r, steps = 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, r, terminated, truncated, info = env.step(int(action))
        total_r += r; steps += 1
    return total_r, steps, info["coverage"], terminated


def evaluate_flat(model_path="models/farm_ppo", n_episodes=50):
    env = FarmEnv(n_beds=4, field_height=8)
    model = MaskablePPO.load(model_path)
    results = [run_flat_episode(model, env, ep) for ep in range(n_episodes)]
    rewards, steps, covs, succs = zip(*results)
    print(f"\n[Step 1 - Flat PPO] n={n_episodes}")
    print(f"  Success:   {np.mean(succs):.1%}")
    print(f"  Reward:    {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Coverage:  {np.mean(covs):.1%}")
    print(f"  Steps:     {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    return {"rewards": rewards, "steps": steps, "covs": covs, "succs": succs, "label": "Step1 Flat PPO"}


# ── Step 2 hierarchical ───────────────────────────────────────────────

def run_hl_episode(hl_model, ll_model, seed):
    env = HighLevelFarmEnv(ll_model, n_beds=4, field_height=8)
    obs, _ = env.reset(seed=seed)
    total_r, total_steps, visits = 0.0, 0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = hl_model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(int(action))
        total_r += r
        total_steps += info["steps_for_lane"]
        visits += 1
    return total_r, total_steps, info["coverage"], terminated, visits


def evaluate_hierarchical(
    ll_path="models/lane_executor",
    hl_path="models/high_level",
    n_episodes=50,
):
    ll_model = MaskablePPO.load(ll_path)
    hl_model = PPO.load(hl_path)
    results = [run_hl_episode(hl_model, ll_model, ep) for ep in range(n_episodes)]
    rewards, steps, covs, succs, visits = zip(*results)
    print(f"\n[Step 2 - Hierarchical RL] n={n_episodes}")
    print(f"  Success:      {np.mean(succs):.1%}")
    print(f"  Reward:       {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Coverage:     {np.mean(covs):.1%}")
    print(f"  Total Steps:  {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"  Lane visits:  {np.mean(visits):.1f}")
    return {"rewards": rewards, "steps": steps, "covs": covs, "succs": succs, "label": "Step2 Hierarchical"}


# ── Comparison plot ───────────────────────────────────────────────────

def plot_comparison(flat, hier):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Step 1 (Flat PPO) vs Step 2 (Hierarchical RL)", fontsize=13, fontweight="bold")

    for ax, key, xlabel, title in [
        (axes[0], "steps",   "Steps",          "Completion Steps"),
        (axes[1], "covs",    "Coverage Rate",   "Field Coverage"),
        (axes[2], "rewards", "Cumulative Reward", "Episode Reward"),
    ]:
        ax.hist(flat[key],  bins=15, alpha=0.6, color="steelblue",  label=flat["label"],  edgecolor="white")
        ax.hist(hier[key],  bins=15, alpha=0.6, color="coral",      label=hier["label"],  edgecolor="white")
        ax.axvline(np.mean(flat[key]), color="blue",  linestyle="--", lw=2, label=f"mean={np.mean(flat[key]):.1f}")
        ax.axvline(np.mean(hier[key]), color="red",   linestyle="--", lw=2, label=f"mean={np.mean(hier[key]):.1f}")
        ax.set_title(title); ax.set_xlabel(xlabel); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results_comparison.png", dpi=140, bbox_inches="tight")
    print("\nSaved: results_comparison.png")

    # Print step reduction
    flat_steps = np.mean(flat["steps"])
    hier_steps = np.mean(hier["steps"])
    reduction = (flat_steps - hier_steps) / flat_steps * 100
    print(f"\nStep reduction: {flat_steps:.1f} → {hier_steps:.1f} ({reduction:+.1f}%)")


if __name__ == "__main__":
    flat = evaluate_flat(n_episodes=50)
    hier = evaluate_hierarchical(n_episodes=50)
    plot_comparison(flat, hier)
```

- [ ] **Step 2: 평가 실행**

```bash
python evaluate_hierarchical.py 2>/dev/null
```

Expected: Step 1 vs Step 2 지표 출력, `results_comparison.png` 생성.

- [ ] **Step 3: 모델 파일 + 결과 이미지 확인**

```bash
ls -lh results_comparison.png
```

Expected: 파일 존재.

- [ ] **Step 4: Commit**

```bash
git add train_hierarchical.py evaluate_hierarchical.py
git add -f results_comparison.png models/lane_executor.zip models/high_level.zip
git commit -m "feat: hierarchical evaluation with Step1 vs Step2 comparison"
```

---

## 완료 기준 체크리스트

- [ ] `pytest tests/ -v` → 63개 이상 PASS
- [ ] `models/lane_executor.zip` 존재
- [ ] `models/high_level.zip` 존재
- [ ] `python evaluate_hierarchical.py` → Step 2 평균 스텝 ≤ 110 (Step 1 대비 25% 이상 감소)
- [ ] `results_comparison.png` 생성
- [ ] 보고서(report.md) Step 2 결과 섹션에 실제 수치 기재
