# 강화학습 프로젝트 보고서
## 정밀농업 자율 로봇을 위한 커스텀 강화학습 환경 설계 및 적용

---

## 1. 문제 정의

### 1.1 연구 배경

스마트팜 및 정밀농업 분야에서 자율 이동 로봇을 이용한 작물 모니터링 및 관리는 핵심 기술로 부상하고 있다. 광범위한 농경지를 인력으로 순찰하는 것은 비효율적이며, 해충 발생이나 수확 적기를 놓치는 문제가 발생할 수 있다. 본 프로젝트는 강화학습(Reinforcement Learning)을 활용하여 자율 농업 로봇이 필드를 효율적으로 순회하며 작물 상태를 파악하고 적절한 조치를 취하는 정책을 학습하는 환경을 설계하였다.

### 1.2 해결 문제

**자율 농업 로봇의 전체 필드 커버리지 및 작물 상태 기반 조치 최적화**

구체적으로 다음 세 가지 조건을 동시에 만족하는 정책을 학습한다:

1. **충돌 없는 이동**: 작물 셀·벽과 충돌하지 않고 경로만 이동
2. **전체 예찰(Coverage)**: 모든 작물 셀을 빠짐없이 스캔
3. **상태 기반 조치**: 예찰 결과에 따라 정상/수확/방제 구분 후 적절한 액션 수행

### 1.3 ROS2 기반 시스템 아키텍처 (설계 관점)

실제 농업 로봇 배포를 가정하면 아래 5개 ROS2 노드로 시스템이 분리된다. 시뮬레이션에서는 Gymnasium 환경이 이를 통합 구현한다.

```
[SceneObserverNode] ──/farm/observation──► [RLAgentNode] ──/farm/cmd_action──►[ActionExecutorNode]
                   ──/farm/action_mask──►                                              │
                                                                                /farm/action_result
[EpisodeManagerNode] ◄──/farm/episode_info──[RewardCalculatorNode]◄────────────────────┘
        │
  /farm/reset (Service)
```

| ROS2 노드 | Gymnasium 구현 |
|-----------|---------------|
| SceneObserverNode | `FarmEnv._get_obs()`, `action_masks()` |
| RLAgentNode | SB3 MaskablePPO 정책 |
| ActionExecutorNode | `FarmEnv.step()` 액션 처리 |
| RewardCalculatorNode | `FarmEnv.step()` 보상 계산 |
| EpisodeManagerNode | `FarmEnv.reset()` + 종료 조건 |

---

## 2. 강화학습 환경 설계

### 2.1 맵 구조

행 기반(row-based) 격자 맵. 작물 열(C)과 수직 주행 레인(P)이 교차 배치되며, 상·하단에 헤드랜드(수평 통로)가 연결된다.

```
col:  0 1 2 3 4 5 6 7 8
row0: W W W W W W W W W   ← 벽
row1: W P P P P P P P W   ← 상단 헤드랜드 (에이전트 시작: col 2)
row2: W C P C P C P C W   ← 필드 행
row3: W C P C P C P C W     (C=작물, P=레인)
...
row8: W P P P P P P P W   ← 하단 헤드랜드
row9: W W W W W W W W W   ← 벽
```

| 항목 | 값 |
|------|-----|
| 맵 크기 | 10 × 9 (H × W) |
| 작물 셀 수 | 24개 |
| 레인(경로) 셀 수 | 32개 |
| 에이전트 시작 위치 | (1, 2) — 상단 헤드랜드, 첫 번째 레인 |

> **맵 고정 정책**: 레이아웃(P/C/W 배치)은 에피소드 간 고정. 각 C 셀의 상태(정상/수확/방제)는 에피소드 시작 시 랜덤 재배정 → 매 에피소드가 다른 문제로 변화하여 일반화된 정책 학습 유도.

### 2.2 상태 공간 (Observation Space)

4채널 그리드를 flattened vector로 표현: **Box(0, 1, shape=(4 × H × W,))**

| 채널 | 내용 | 정규화 |
|------|------|--------|
| ch0 | 맵 구조 (0=통로, 1=작물, 2=벽) | ÷ 2 |
| ch1 | 에이전트 위치 원-핫 | 0 or 1 |
| ch2 | 예찰 완료 마스크 | 0 or 1 |
| ch3 | 예찰 결과 (0~5) | ÷ 5 |

ch3 값 정의:

| 값 | 의미 |
|----|------|
| 0 | 미예찰 (상태 미지) |
| 1 | 정상 확인 완료 |
| 2 | 수확 필요 (Pending) |
| 3 | 방제 필요 (Pending) |
| 4 | 수확 완료 |
| 5 | 방제 완료 |

> **두 계층 상태 설계**: `_true_states` (은닉, 에이전트 불가시)와 `crop_states` (공개, 예찰 후 reveal). 에이전트는 반드시 Scout 액션을 수행해야 작물 상태를 알 수 있어 탐색 유인이 발생한다.

### 2.3 행동 공간 (Action Space) + Action Masking

**Discrete(7)** — MaskablePPO의 Action Masking 적용으로 무효 액션을 물리적으로 차단.

| 인덱스 | 액션 | 마스킹 조건 |
|--------|------|------------|
| 0 | 이동 (상) | 목적지가 작물/벽 |
| 1 | 이동 (하) | 목적지가 작물/벽 |
| 2 | 이동 (좌) | 목적지가 작물/벽 |
| 3 | 이동 (우) | 목적지가 작물/벽 |
| 4 | 예찰 (Scout) | 인접 미예찰 C 셀 없음 |
| 5 | 수확 (Harvest) | 인접 수확 대기 셀 없음 |
| 6 | 방제 (Pest Control) | 인접 방제 대기 셀 없음 |

**Scout 동작**: 에이전트 현재 위치 기준 인접 4방향 C 셀 전부의 숨겨진 상태를 동시 reveal.  
**정상 셀**: Scout 시 자동 처리 완료 — 별도 액션 불필요.

### 2.4 보상 함수 (Reward Function)

| 이벤트 | 보상 | 설계 근거 |
|--------|------|----------|
| 매 스텝 | −0.1 | 불필요한 배회 억제 |
| 충돌 시도 | −2.0 | 안전 항법 강제 |
| 신규 셀 예찰 | +1.0 | 탐색 유인 |
| 정상 확인 | +0.5 | 예찰 추가 인센티브 |
| 수확 성공 | +10.0 | 핵심 목표 |
| 방제 성공 | +8.0 | 핵심 목표 |
| 전체 완료 | +20.0 | 커버리지 완성 유도 |

### 2.5 에피소드 종료 조건

```python
# 성공 종료 (terminated)
done = all(crop_states[c] in {1, 4, 5} for c in all_crop_cells)
# 즉, 모든 C 셀이 정상확인(1) 또는 수확완료(4) 또는 방제완료(5)

# 시간 초과 (truncated)
done = step_count >= max_steps   # max_steps = H × W × 3 = 270
```

### 2.6 작물 상태 초기 분포

| 상태 | 확률 |
|------|------|
| 정상 (조치 불필요) | 60% |
| 수확 필요 | 25% |
| 방제 필요 | 15% |

---

## 3. 알고리즘 선택 및 적용

### 3.1 알고리즘: MaskablePPO

본 환경은 **상황에 따라 유효한 액션이 달라지는 동적 액션 마스킹** 문제이므로, 표준 PPO를 확장한 **MaskablePPO** (sb3-contrib)를 선택하였다.

**PPO를 선택한 이유**:
- 이산 행동 공간에서 안정적 학습
- On-policy 방법으로 샘플 효율 적절
- SB3의 표준 구현으로 재현성 보장

**Action Masking이 필요한 이유**:
- 단순 PPO는 무효 액션(벽으로 이동, 예찰 전 수확 등)도 선택 가능 → 음의 보상으로만 억제
- Action Masking은 소프트맥스 전 로짓에 −∞ 마스크를 적용 → 무효 액션 선택 자체를 원천 차단
- 특히 탐색 초기에 불필요한 샘플 낭비 방지 → 수렴 가속

### 3.2 정책 네트워크

```
MlpPolicy: [360-dim flat obs] → [256, 256 FC] → [7-dim logit] → Masked Softmax → Action
```

관측값 360 = 4채널 × 10 × 9 (flattened). CnnPolicy 대신 MlpPolicy를 사용한 이유: SB3 기본 CnnPolicy는 Atari(84×84) 기준 설계로, 10×9 소형 그리드에서 conv stride/kernel이 맞지 않아 학습 불안정 가능성.

### 3.3 하이퍼파라미터

| 파라미터 | 값 |
|----------|----|
| total_timesteps | 500,000 |
| n_steps | 2,048 |
| batch_size | 64 |
| n_epochs | 10 |
| learning_rate | 3×10⁻⁴ |
| gamma (할인율) | 0.99 |
| ent_coef (엔트로피 계수) | 0.01 |
| 병렬 환경 수 | 4 (DummyVecEnv) |

---

## 4. 실험 과정

### 4.1 구현 구조

```
rlproject/
├── env/
│   ├── constants.py      # 상수 정의 (셀 타입, 상태, 보상)
│   ├── map_generator.py  # 맵 생성 + 작물 상태 초기화
│   └── farm_env.py       # Gymnasium 커스텀 환경
├── train.py              # MaskablePPO 학습 스크립트
├── evaluate.py           # 평가 및 시각화
├── record_gif.py         # 에이전트 동작 영상 녹화
└── models/farm_ppo.zip   # 학습된 모델
```

**테스트 커버리지**: 36개 단위 테스트 전체 통과 (pytest)
- 맵 구조 검증, 액션 마스킹 로직, step() 동작, 보상 계산, 종료 조건, Gymnasium API 준수 포함

### 4.2 학습 과정

- **학습 환경**: FarmEnv(n_lanes=3, field_height=6), 4개 병렬 환경
- **총 학습 스텝**: 500,000 timesteps
- **최종 ep_rew_mean**: 약 138 (학습 말기 기준)
- **수렴 패턴**: 초기에는 랜덤 탐색 → 약 100k 스텝 이후 커버리지 패턴 학습 → 200k 이후 안정적 수렴

---

## 5. 결과 분석

### 5.1 정량적 평가 (50 에피소드)

| 지표 | 결과 |
|------|------|
| **성공률** | **100% (50/50)** |
| **평균 누적 보상** | **139.4 ± 19.7** |
| **평균 필드 커버리지** | **100%** |
| **평균 완료 스텝** | **37.8 ± 1.8** |
| 최소/최대 스텝 | 34 / 42 |

### 5.2 결과 분석

**Action Masking의 효과**: 무효 액션 차단으로 초기 탐색 단계에서 충돌 패널티 없이 효율적 정책 학습. 학습된 정책은 충돌 시도를 전혀 하지 않음.

**자발적 Boustrophedon 패턴 학습**: 에이전트는 명시적으로 지그재그 경로를 학습하도록 설계되지 않았으나, 보상 구조(스텝 패널티 + 커버리지 완료 보너스)만으로 스스로 최적에 가까운 Boustrophedon(경작 지그재그) 패턴을 학습하였다. 이는 강화학습의 보상 기반 정책 발견 능력을 잘 보여준다.

**두 계층 상태 설계의 유효성**: 은닉 true state와 공개 observed state를 분리함으로써 부분 관측(POMDP) 환경을 자연스럽게 구현. 에이전트는 Scout 액션을 반드시 수행해야 수확/방제 대상을 파악할 수 있어 탐색과 활용(Exploration-Exploitation)의 균형을 능동적으로 학습.

**수확/방제 정확도**: Action Masking에 의해 예찰 전 수확/방제 액션이 불가능하므로, 예찰 이후 정확한 상태 기반 액션만 선택됨. 오분류에 의한 실패 없음.

### 5.3 평가 결과 그래프

*(results_evaluation.png 참조)*
- **좌**: 누적 보상 분포 — 평균 139.4, 표준편차 19.7로 안정적
- **중**: 종료 유형 — 50/50이 정상 종료(terminated), 시간 초과 없음
- **우**: 완료 스텝 분포 — 34~42 스텝의 좁은 범위에서 일관된 완료

---

## 6. 결론

본 프로젝트는 정밀농업 자율 로봇 시나리오를 강화학습 문제로 정의하고, ROS2 노드 아키텍처 기반의 Gymnasium 커스텀 환경을 설계·구현하였다. MaskablePPO 알고리즘과 Action Masking을 결합하여 500k 스텝 학습 후 **50 에피소드 전체에서 100% 성공률과 100% 커버리지**를 달성하였다.

에이전트는 보상 설계만으로 스스로 최적에 가까운 경로 탐색 전략(Boustrophedon 패턴)을 습득하였으며, 이는 복잡한 경로계획 알고리즘 없이도 RL이 실용적 농업 로봇 정책을 학습할 수 있음을 보여준다.

---

## 참고

- Gymnasium: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- sb3-contrib (MaskablePPO): https://sb3-contrib.readthedocs.io/
- Boustrophedon Coverage: Choset, H. (2001). Coverage for robotics – A survey of recent results.
