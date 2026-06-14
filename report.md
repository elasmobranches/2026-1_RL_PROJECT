# 강화학습 프로젝트 보고서
## 정밀농업 자율 로봇을 위한 커스텀 강화학습 환경 설계 및 적용

**학과**: AI로봇학과  
**제출일**: 2026-06-15  
**사용 언어**: Python 3.10 (Gymnasium, Stable-Baselines3, sb3-contrib)

---

## 1. 문제 정의

### 1.1 연구 배경 및 동기

스마트팜 관련 연구를 하다 보면 실제로 가장 귀찮은 게 작물 상태 확인이다. 드론이나 카메라로 전체를 찍는 방법도 있지만, 해충이 생긴 건 가까이서 봐야 알 수 있고 수확 적기도 직접 확인해야 정확하다. 그래서 자율 이동 로봇이 직접 돌아다니면서 필드를 순찰하고 조치까지 취하는 시스템을 강화학습으로 구현해보기로 했다.

### 1.2 강화학습 문제로의 정식화

이 문제는 다음과 같은 이유로 강화학습에 적합하다.

첫째, **순차적 의사결정**이다. 현재의 이동 선택이 이후 도달 가능한 영역과 잔여 작업량을 바꾸므로, 매 스텝의 행동이 미래 보상에 영향을 준다. 이는 단순 분류·회귀가 아닌 누적 보상 최대화 문제의 전형적 형태다.

둘째, **부분 관측(POMDP)**이다. 로봇은 예찰(Scout) 행동을 하기 전에는 작물의 실제 상태(수확 대기/방제 대기/정상)를 알 수 없다. "정보를 얻는 행동(예찰)"과 "정보를 활용하는 행동(조치)"이 분리되어 있어 관측이 불완전하다. 이때 에이전트(또는 계층적 구조에서는 상위 정책)가 어느 순서로 탐색하고 조치할지를 스스로 결정해야 한다는 점이 일반 규칙 기반 시스템보다 RL이 자연스러운 이유다.

셋째, **경로 최적화의 조합론적 복잡성**이다. 5개 레인 방문 순서만 해도 최소 120가지(5!) 이상이며, 매 에피소드마다 작물 상태가 랜덤하게 바뀌므로 고정 규칙으로는 상황에 맞는 최적 대응이 불가능하다. "수확 대기 발견 → 수확" 같은 개별 행동은 규칙화할 수 있어도, "지금 어느 방향으로 이동할지"는 현재 작물 분포와 미래 경로를 동시에 고려해야 하므로 보상 신호로 학습하는 게 적합하다.

### 1.3 해결 목표

**자율 농업 로봇의 전체 필드 커버리지 및 작물 상태 기반 조치 최적화**

| 조건 | 설명 |
|------|------|
| 충돌 없는 이동 | 작물 구역·벽과 충돌하지 않고 주행 레인만 이동 |
| 전체 예찰 | 모든 작물 셀을 빠짐없이 스캔 |
| 상태 기반 조치 | 예찰 결과에 따라 정상 확인/수확/방제 구분 후 조치 |

### 1.4 ROS2 기반 시스템 아키텍처

실제 로봇 배포를 가정한 설계. 시뮬레이션에서는 Gymnasium 환경이 5개 노드를 통합 구현한다.

```
[SceneObserverNode]──/farm/observation──►[RLAgentNode]──/farm/cmd_action──►[ActionExecutorNode]
                  ──/farm/action_mask──►                                           │
                                                                         /farm/action_result
[EpisodeManagerNode]◄──/farm/episode_info──[RewardCalculatorNode]◄─────────────────┘
```

---

## 2. 강화학습 환경 설계

> **이 섹션은 Step 1(FarmEnv) 기반 설계를 기준으로 설명한다.** Step 2/3 계층형과 Step 4 연속 환경은 이 설계를 확장·수정하며, 각 단계에서 변경 사항을 별도 기술한다.

### 2.1 맵 구조

현실 과수원·포도밭 구조를 모델링한 **2열 재배단** 격자 맵.

![맵 구조](map_structure.png)

```
최종 맵 (n_beds=4, field_height=8): H=12, W=15

col: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
r0:  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W
r1:  W  P  P  P  P  P  P  P  P  P  P  P  P  P  W   ← 상단 헤드랜드
r2:  W  P  C  C  P  C  C  P  C  C  P  C  C  P  W   ← 필드 행 (r2~r9)
r10: W  P  P  P  P  P  P  P  P  P  P  P  P  P  W   ← 하단 헤드랜드
r11: W  W  W  W  W  W  W  W  W  W  W  W  W  W  W

레인 열(col): 1, 4, 7, 10, 13  /  작물 셀: 64개
```

레인에서 인접한 **안쪽 열만** 스캔 가능하며, 바깥쪽 열은 반대편 레인에서만 접근 가능하다. 전체 커버리지를 위해 5개 레인 모두 방문해야 한다.

| 항목 | 값 |
|------|-----|
| 맵 크기 | 12 × 15 |
| 작물 셀 수 | 64개 |
| 주행 레인 수 | 5개 |
| 에이전트 시작 | (1, 1) — 상단 헤드랜드 |

### 2.2 상태 공간

**이산 환경 (Step 1 기준)**: `Box(0, 1, shape=(720,))` — 4채널 × 12 × 15

| 채널 | 내용 | 정규화 |
|------|------|--------|
| ch0 | 맵 구조 (0=통로, 1=작물, 2=벽) | ÷2 |
| ch1 | 에이전트 위치 one-hot | 0/1 |
| ch2 | 예찰 완료 마스크 | 0/1 |
| ch3 | 예찰 결과 (0=미지~5=방제완료) | ÷5 |

> **POMDP 설계**: `_true_states`(은닉)와 `crop_states`(공개)를 분리. Scout 전까지 ch3=0 — 탐색 유인이 구조적으로 발생한다.

> **Step 2/3 계층형**: ch4(목표 레인 마스크) 추가 → 5채널 × 900-dim. High-level이 지정한 목표 레인 위치를 Low-level에 전달하는 채널이다.

**연속 환경 (Step 4)**: `Box(-1, 1, shape=(28,))`

| 구성 | 차원 | 내용 |
|------|------|------|
| robot_pos + heading | 4 | 위치(x,y) + 이동 방향(cos,sin) |
| nav_flags | 4 | 4방향 이동 가능 여부 (0/1) |
| top-5 nearest crops | 20 | (dx, dy, revealed, state) × 5 |

### 2.3 행동 공간

**이산 환경**: `Discrete(7)` + Action Masking

| 인덱스 | 액션 | 마스킹 조건 |
|--------|------|------------|
| 0~3 | 이동 (상/하/좌/우) | 목적지가 작물/벽 |
| 4 | 예찰 (Scout) | 인접 미예찰 셀 없음 |
| 5 | 수확 (Harvest) | 인접 수확대기 셀 없음 |
| 6 | 방제 (Pest Control) | 인접 방제대기 셀 없음 |

Action Masking은 소프트맥스 전 로짓에 −∞를 적용해 무효 액션 선택 확률을 0으로 만든다. 초기 탐색 단계에서 샘플 낭비를 막아 수렴을 가속시킨다.

**연속 환경 (Step 4)**: `Box(-1, 1, shape=(2,))` — 속도 벡터 (vx, vy)

### 2.4 보상 함수

**이산 환경 (Step 1 기준)**:

| 이벤트 | 보상 | 설계 근거 |
|--------|------|----------|
| 매 스텝 | −0.1 | 배회 억제 |
| 충돌 시도 | −2.0 | 안전 항법 강제 |
| 신규 셀 예찰 | +1.0 | 탐색 유인 |
| 정상 확인 | +0.5 | 예찰 인센티브 |
| 수확 성공 | +10.0 | 핵심 목표 |
| 방제 성공 | +8.0 | 핵심 목표 |
| 전체 완료 | +20.0 | 커버리지 완성 유도 |

> **Step 2/3 계층형 추가**: 레인 완료 시 +10.0 보너스.

**연속 환경 (Step 4)**: 이산 대비 스케일 조정 (R_STEP=−0.05, R_COLLISION=−1.0, R_SCOUT_NEW=+1.5) + Potential-based shaping.

Potential-based shaping: `F = γ·φ(s') − φ(s)`, `φ(s) = −min_dist × 0.3`

단순 근접 보상은 에이전트가 한 지점에 정착해 보상을 누적하는 함정을 유발한다. Ng et al.(1999)의 잠재 함수 차분 방식은 최적 정책 불변성을 보장하면서 학습 신호만 조밀하게 만들어 준다.

### 2.5 에피소드 종료 조건

```python
# 성공 종료: 모든 작물 셀이 처리 완료 상태
done = all(crop_states[c] in {1, 4, 5} for c in all_crop_cells)

# 시간 초과
done = step_count >= max_steps
```

| 모델 | 성공 종료 기준 | max_steps |
|------|--------------|-----------|
| Step 1 FarmEnv | 전체 작물 완료 | 540 |
| Step 2/3 LaneExecutorEnv | 목표 레인 작물 완료 | 180 (레인당) |
| Step 4 연속 환경 | 전체 작물 완료 | 1200 |

### 2.6 작물 상태 초기 분포

| 상태 | 확률 | 의미 |
|------|------|------|
| 정상 (1) | 60% | 조치 불필요 |
| 수확 필요 (2) | 25% | 성숙 과일 수확 |
| 방제 필요 (3) | 15% | 해충 방제 |

매 에피소드마다 랜덤 초기화되므로 에이전트는 고정 패턴을 암기할 수 없고, 예찰 결과에 따라 동적으로 대응해야 한다.

---

## 3. 알고리즘 선택 및 적용

처음엔 단순 Flat RL로 시작했다가 한계가 보일 때마다 계층 구조 → 거리 인식 → 연속 환경 순으로 발전시켰다.

### 3.1 Step 1: MaskablePPO

이산 행동 공간 + 동적 Action Masking이 동시에 필요한 문제. sb3-contrib의 MaskablePPO가 두 조건을 모두 지원한다. 표준 PPO 대비 Action Masking으로 초기 탐색 효율이 크게 향상된다.

| 파라미터 | 값 |
|----------|----|
| n_envs | 16 |
| learning_rate | 3×10⁻⁴ |
| total_timesteps | 1,500,000 |

### 3.2 Step 2/3: Hierarchical RL

Flat RL은 "어느 레인 방문"(전역)과 "레인 내 행동"(국소)이 하나의 정책에 혼재된다. 계층 분리로 각 수준에 특화된 정책을 학습한다.

```
High-level (PPO): 레인 선택 — obs: [레인 완료율(5), 거리(5)] = 10-dim
       ↓ 목표 레인 지정
Low-level (MaskablePPO): 레인 처리 — obs: 전체 맵 + ch4 = 900-dim
```

**거리 인식 obs의 핵심 역할**: High-level obs에 현재 위치에서 각 레인까지의 거리를 포함함으로써, DQN이 "가장 가까운 미완료 레인"을 선택하는 Greedy 최적 전략을 자동으로 학습하였다.

### 3.3 Step 4: SAC (연속 환경)

이산 격자에서 벗어나 **연속 2D 좌표 + 속도 제어**로 현실적인 로봇 이동을 구현한다. 이산 행동이 없으므로 연속 제어에 특화된 SAC를 선택한다.

SAC의 **최대 엔트로피 프레임워크**는 희박 보상 환경에서 자동으로 탐색 정도를 조절해 TD3·TQC 대비 우월한 성능을 보였다 (Section 5.3 참조).

학습 가속 기법:
- 관측 단순화: 30개 작물 전부 → 가장 가까운 5개 (124-dim → 28-dim)
- nav_flags: 4방향 이동 가능 여부 명시 (충돌 방향 인식)
- 커리큘럼: Level 0(작물 옆 스폰) → Level 2(헤드랜드 정상 스폰)

---

## 4. 실험 과정

### 4.1 코드 구조

```
rlproject/
├── env/
│   ├── farm_env.py               # Step 1 이산 환경
│   ├── map_generator.py          # 2열 재배단 맵 생성
│   ├── continuous_farm_env.py    # Step 4 연속 환경
│   ├── continuous_farm_env_curriculum.py
│   └── hierarchical/
│       ├── lane_executor_env.py  # Step 2/3 Low-level
│       └── high_level_env.py     # Step 2/3 High-level
├── train.py                      # Step 1
├── train_hierarchical.py         # Step 2
├── train_step3.py                # Step 3
├── train_sac_curriculum.py       # Step 4 SAC
└── evaluate*.py / record_gif*.py
```

### 4.2 실행 방법

```bash
pip install -r requirements.txt

python train.py                   # Step 1
python train_hierarchical.py      # Step 2
python train_step3.py             # Step 3
python train_sac_curriculum.py    # Step 4 SAC

python evaluate_step3.py          # Step 1~3 비교
python evaluate_hierarchical.py   # Step 1~2 비교
python record_gif_step3.py        # Step 3 데모
python record_gif_sac_curriculum.py  # Step 4 데모
```

### 4.3 학습 환경

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA RTX 3080 (CUDA 12.4) |
| CPU | 병렬 env 수집 |
| PyTorch | 2.5.1+cu124 |
| Python | 3.10 |

---

## 5. 결과 분석

### 5.1 최종 성능 비교

| 단계 | 알고리즘 | 성공률 | 커버리지 | 평균 스텝 |
|------|----------|--------|---------|---------|
| Step 1 | MaskablePPO (Flat) | 86~96% | 99.8% | 147~211 |
| Step 2 | Hierarchical PPO | 98% | 99.8% | 198 |
| **Step 3** | **Goal+DQN+거리 인식** | **100%** | **100%** | **141 ± 3** |
| Step 4 | SAC + Curriculum | 96.7% | 99.9% | 157 |

*(results_step3_comparison.png, results_evaluation.png 참조)*

### 5.2 핵심 발견

**1. Action Masking이 없으면 이산 환경에서 학습 자체가 안 된다**

RecurrentPPO로 masking 없이 돌려봤더니 커버리지 0.9%로 완전 실패. 무효 액션에 샘플을 낭비하면 수렴이 불가능하다.

**2. 보상 스케일보다 관측 공간 설계가 훨씬 결정적이었다**

스텝 패널티를 −0.1→−0.3으로 강화했더니 성공률이 96%→80%로 떨어졌다. 반면 High-level obs에 거리 정보 하나 추가했더니 90%→100%로 올랐다. "에이전트에게 필요한 정보를 제대로 주는 것"이 보상 튜닝보다 훨씬 중요하다.

**3. 올바른 obs를 주면 RL이 최적 전략을 스스로 발견한다**

거리 정보를 obs에 넣자 DQN이 학습한 정책 = Greedy nearest-lane과 완전히 동일 (141±3 steps, 2.0 visits). 따로 "가까운 것부터 가라"고 가르치지 않았는데 스스로 그 전략을 찾아냈다.

**4. 연속 환경에서는 SAC의 최대 엔트로피가 결정적이었다**

| 알고리즘 | 성공률 | 커버리지 |
|---------|--------|---------|
| TD3 | 0% | 42.3% |
| TQC | 3.3% | 60.0% |
| **SAC** | **96.7%** | **99.9%** |

탐색 능력이 핵심인 환경에서 SAC의 자동 엔트로피 조절이 압도적으로 유리했다. *(results_algo_comparison.png 참조)*

**5. Boustrophedon 패턴의 자발적 학습**

명시적 경로 알고리즘 없이 보상 설계만으로 지그재그 순회 전략을 스스로 학습했다. RL의 정책 발견 능력을 잘 보여주는 결과다.

### 5.3 알고리즘 선택 타당성 검증

SAC vs A2C vs PPO 비교:

*(results_algo_comparison.png 참조)*

이산 환경에서 RecurrentPPO(LSTM)는 Action Masking 미지원으로 실패했고, 연속 환경에서는 SAC가 최우수 성능을 보였다. A2C는 이산·연속 모두에서 PPO보다 낮았으며, 이는 PPO의 클리핑 메커니즘이 학습 안정성에 기여한다는 것을 실험적으로 확인한 결과다.

---

## 6. 시행착오 및 실험 과정

이 섹션은 최종 결과에 이르기까지 시도하고 실패하거나 개선한 과정을 기록한다.

### 6.1 환경 설계 변경

**1열 → 2열 재배단**: 초기 설계는 1열 재배단(24셀)이었다. 실제 농경지에서는 레인에서 안쪽 열만 스캔 가능하고 바깥쪽은 반대편 레인에서 봐야 한다는 점을 반영해 2열 구조(64셀)로 변경했다. 더 현실적인 탐색 문제가 됐다.

### 6.2 Step 패널티 강화 실험 (실패)

Low-level 스텝 패널티를 −0.1→−0.3으로 높이면 에이전트가 목표 레인으로 더 빠르게 이동할 것이라 기대했다. 결과는 성공률 96%→80% 급락. 강한 시간 압박이 레인 완료 실패율을 높였다. 보상 스케일보다 관측 설계가 중요하다는 교훈을 얻었다.

### 6.3 SAC 연속 환경 반복 개선

| 시도 | 변경 사항 | 결과 | 실패 원인 |
|------|-----------|------|----------|
| v1 | 기본 SAC, obs=124-dim | 커버리지 0% | 희박 보상 + 고차원 obs |
| v2 | Potential shaping + 랜덤 시작 | 커버리지 6% | 정착 함정 (proximity 보상) |
| v3 | **nav_flags 추가** | 커버리지 56% | 방향 인식 부재 |
| **v4** | **obs 단순화 + 커리큘럼** | **96.7%** | — |

v1~v2 실패의 핵심: 단순 근접 보상은 로봇이 시작점 근처에서 왔다갔다하며 보상을 누적하는 함정에 빠진다. Potential-based shaping(Ng et al. 1999)으로 교체해 해결했다.

v3에서 nav_flags(4방향 이동 가능 여부) 추가가 결정적이었다. 이전엔 로봇이 작물 구역이 막혀 있다는 사실을 obs로 알 방법이 없어서 계속 충돌을 시도하며 제자리에 고착됐다.

### 6.4 SAC Large Map (부분 성공)

n_beds=4 더 큰 연속 맵에서 SAC를 학습시켰으나 1.5M 스텝으로 커버리지 37%에 그쳤다. 맵이 커질수록 학습 스텝이 비례해 늘어나는 것을 확인했다. 향후 연구 과제.

### 6.5 계층형 연속 환경 실험 (실패)

이산 환경의 계층 구조 성공에 고무돼, 연속 환경에도 계층 구조(SAC/PPO/A2C LL + DQN HL)를 적용해봤다.

| LL 알고리즘 | 성공률 | 커버리지 | 실패 원인 |
|------------|--------|---------|----------|
| SAC | 5% | 41.7% | goal-conditioned 학습 부족 |
| PPO | 0% | 21.7% | 동일 |
| A2C | 0% | 16.7% | 동일 + 불안정 |

실패 원인: 이산 환경 LL은 전체 필드를 처리하면 되지만, 연속 환경 LL은 "지정된 레인으로 이동해서 그 레인만 처리하고 종료"라는 목표 조건부(goal-conditioned) 태스크다. 이산보다 훨씬 어려운 문제인데 1M 스텝으로는 수렴이 안 됐다. 충분한 학습(3M+) 또는 별도의 커리큘럼이 필요하다.

부가적으로 VecNormalize 불일치 버그도 발견했다: SAC LL을 VecNormalize로 학습시키면 HL env가 raw obs를 직접 넘길 때 분포가 달라 정책이 망가진다. VecNormalize 제거로 해결.

### 6.6 코드 버그 발견 및 수정

리뷰 중 발견한 버그들:

| 버그 | 파일 | 영향 | 수정 |
|------|------|------|------|
| `_goal_reached` 레인 전환 시 미리셋 | high_level_env.py | 2번째 이후 레인 goal bonus 누락 | 전환마다 False 리셋 |
| `_prev_potential` __init__ 미초기화 | continuous_farm_env.py | reset() 전 step() 호출 시 crash | 0.0으로 초기화 |
| _handle_move 범위 체크 없음 | farm_env.py | 맵 경계 외 접근 가능 | bounds check 추가 |

---

## 7. 결론

### 7.1 전체 성능 요약

| 단계 | 알고리즘 | 성공률 | 커버리지 | 핵심 기여 |
|------|----------|--------|---------|----------|
| Step 1 | MaskablePPO | 96% | 99.8% | Action Masking, Boustrophedon 자발 학습 |
| Step 2 | Hierarchical PPO | 98% | 99.8% | 계층 분리, 암묵적 다중 레인 커버리지 |
| **Step 3** | **Goal+DQN** | **100%** | **100%** | 거리 인식 obs, Greedy 최적 정책 자동 발견 |
| Step 4 | SAC+Curriculum | 96.7% | 99.9% | 연속 공간, 최대 엔트로피 탐색 |

### 7.2 관측 공간 설계의 중요성

이번 프로젝트에서 가장 크게 얻은 교훈은 **보상 함수 설계보다 관측 공간 설계가 훨씬 결정적**이라는 점이다. 패널티 스케일을 바꾸는 것보다 에이전트에게 필요한 정보를 제대로 주는 것이 수렴 속도와 최종 성능 모두에서 훨씬 영향이 컸다.

### 7.3 실제 로봇 적용 방향

```
시뮬레이션(PPO/Step 3)으로 pre-training
       ↓
실제 로봇(SAC)으로 fine-tuning
       ↓
ROS2 노드로 배포 (/farm/cmd_vel → 속도 명령)
```

### 7.4 향후 연구

- 다중 로봇 협력(MARL): MAPPO 기반 분할 예찰
- Hierarchical SAC: 더 많은 학습 스텝(3M+)으로 연속 계층형 재시도
- 실제 ROS2 배포: Gymnasium → 실제 로봇 연결

---

## 영상 클립

| 파일 | 내용 |
|------|------|
| `agent_demo.gif` | Step 1 MaskablePPO |
| `agent_demo_hierarchical.gif` | Step 2 Hierarchical PPO |
| `agent_demo_step3.gif` | Step 3 Goal+DQN (100% 성공) |
| `agent_demo_sac_curriculum.gif` | Step 4 SAC Curriculum |
| `agent_demo_greedy.gif` | 비교: Greedy nearest-lane |
| `agent_demo_td3.gif` | 비교: TD3 |
| `agent_demo_tqc.gif` | 비교: TQC |

---

## 참고 문헌

- Stable-Baselines3: Raffin et al. (2021). JMLR.
- MaskablePPO: Huang & Ontañón (2022).
- SAC: Haarnoja et al. (2018). ICML.
- TD3: Fujimoto et al. (2018). ICML.
- TQC: Kuznetsov et al. (2020). ICML.
- Potential-based Shaping: Ng, Russell & Bartlett (1999). ICML.
- Hierarchical RL: Nachum et al. (2018). NeurIPS.
- Boustrophedon Coverage: Choset (2001). Ann. Math. Artif. Intell.
