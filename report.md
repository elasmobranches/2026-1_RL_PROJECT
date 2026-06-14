# 강화학습 프로젝트 보고서
## 정밀농업 자율 로봇을 위한 커스텀 강화학습 환경 설계 및 적용

**학과**: AI로봇학과  
**제출일**: 2026-06-15  
**사용 언어**: Python 3.10 (Gymnasium, Stable-Baselines3, sb3-contrib)

---

## 1. 해결하고자 하는 문제 정의

### 1.1 연구 배경 및 동기

스마트팜 관련 연구를 하다 보면 실제로 가장 귀찮은 게 작물 상태 확인이다. 드론이나 카메라로 전체를 찍는 방법도 있지만, 해충이 생긴 건 가까이서 봐야 알 수 있고 수확 적기도 직접 확인해야 정확하다. 그래서 자율 이동 로봇이 직접 돌아다니면서 필드를 순찰하고 조치까지 취하는 시스템을 강화학습으로 구현해보기로 했다.

### 1.2 강화학습 문제로의 정식화

이 문제는 다음과 같은 이유로 강화학습에 적합하다.

첫째, **순차적 의사결정**이다. 현재의 이동 선택이 이후 도달 가능한 영역과 잔여 작업량을 바꾸므로, 매 스텝의 행동이 미래 보상에 영향을 준다. 이는 단순 분류·회귀가 아닌 누적 보상 최대화 문제의 전형적 형태다.

둘째, **부분 관측(POMDP)**이다. 로봇은 예찰(Scout) 행동을 하기 전에는 작물의 실제 상태(수확 대기/방제 대기/정상)를 알 수 없다. "정보를 얻는 행동(예찰)"과 "정보를 활용하는 행동(조치)"이 분리되어 있어 관측이 불완전하다. 이때 에이전트(또는 계층적 구조에서는 상위 정책)가 어느 순서로 탐색하고 조치할지를 스스로 결정해야 한다는 점이 일반 규칙 기반 시스템보다 RL이 자연스러운 이유다.

셋째, **경로 최적화의 조합론적 복잡성**이다. 5개 레인 방문 순서만 해도 120가지이며, 매 에피소드마다 작물 상태가 랜덤하게 바뀌므로 고정 규칙으로는 상황에 맞는 최적 대응이 불가능하다. "수확 대기 발견 → 수확" 같은 개별 행동은 규칙화할 수 있어도, "지금 어느 방향으로 이동할지"는 현재 작물 분포와 미래 경로를 동시에 고려해야 하므로 보상 신호로 학습하는 게 적합하다.

본 연구는 이 문제를 커스텀 Gymnasium 환경으로 구현하고, 여러 알고리즘 중 RL 에이전트가 어떤 정책을 학습하는지, 그 정책을 좌우하는 환경 설계 요소가 무엇인지를 탐구한다.

### 1.3 문제 정의

**자율 농업 로봇의 전체 필드 커버리지(Coverage) 및 작물 상태 기반 조치 최적화**

다음 세 조건을 동시에 만족하는 정책 학습:

| 조건 | 설명 |
|------|------|
| 충돌 없는 이동 | 작물 구역·벽과 충돌하지 않고 주행 레인만 이동 |
| 전체 예찰(Coverage) | 모든 작물 셀을 빠짐없이 스캔(예찰) |
| 상태 기반 조치 | 예찰 결과에 따라 **정상(확인)/수확/방제** 구분 후 적절한 액션 |

### 1.3 ROS2 기반 시스템 아키텍처 (실제 로봇 배포 관점)

```
[SceneObserverNode]──/farm/observation──►[RLAgentNode]──/farm/cmd_action──►[ActionExecutorNode]
                  ──/farm/action_mask──►                                           │
                                                                         /farm/action_result
[EpisodeManagerNode]◄──/farm/episode_info──[RewardCalculatorNode]◄─────────────────┘
```

시뮬레이션에서는 Gymnasium 환경이 5개 ROS2 노드를 통합 구현한다.

---

## 2. 강화학습 환경 설계

> **이 섹션은 Step 1(FarmEnv) 기반 설계를 기준으로 설명한다.** Step 2/3 계층형 변형과 Step 4 연속 환경은 이 설계를 확장·수정하며, 각 단계 섹션에서 변경 사항을 별도로 기술한다.

### 2.1 맵 구조 (이산 격자 환경)

현실 과수원·포도밭 구조를 모델링한 **2열 재배단(two-column bed)** 격자 맵.

```
최종 맵 (n_beds=4, field_height=8): H=12, W=15

col: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
r0:  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W   ← 벽
r1:  W  P  P  P  P  P  P  P  P  P  P  P  P  P  W   ← 상단 헤드랜드
r2:  W  P  C  C  P  C  C  P  C  C  P  C  C  P  W   ← 필드 행
...                                                   (r2~r9 동일)
r10: W  P  P  P  P  P  P  P  P  P  P  P  P  P  W   ← 하단 헤드랜드
r11: W  W  W  W  W  W  W  W  W  W  W  W  W  W  W   ← 벽

P=주행 레인(통로), C=작물 셀, W=벽
레인 열(col): 1, 4, 7, 10, 13  (5개)
```

![맵 구조](map_structure.png)

**2열 재배단의 의미**: 레인에서 바라봤을 때 인접한 **안쪽 열만** 스캔 가능하며, 바깥쪽 열은 반대편 레인에서만 접근 가능. 전체 커버리지를 위해 5개 레인 모두 방문 필요.

| 항목 | 값 |
|------|-----|
| 맵 크기 | **12 × 15 (H × W)** |
| 작물 셀 수 | **64개** |
| 주행 레인 수 | **5개** |
| 에이전트 시작 | (1, 1) — 상단 헤드랜드, 첫 번째 레인 |
| max_steps | 540 (H×W×3) |

### 2.2 상태 공간 (Observation Space)

**이산 환경**: `Box(0, 1, shape=(4×H×W,))` = 720-dim flat vector

| 채널 | 내용 | 정규화 | 설명 |
|------|------|--------|------|
| ch0 | 맵 구조 | ÷2 | 0=통로, 1=작물, 2=벽 |
| ch1 | 에이전트 위치 | 0/1 | 현재 위치 one-hot |
| ch2 | 예찰 완료 마스크 | 0/1 | 예찰한 셀=1 |
| ch3 | 예찰 결과 | ÷5 | 0=미지, 1=정상, 2=수확대기, 3=방제대기, 4=수확완료, 5=방제완료 |

> **부분 관측 설계(POMDP)**: `_true_states`(은닉)와 `crop_states`(공개)를 분리. 에이전트는 Scout 액션 없이는 작물 상태를 알 수 없어 탐색 유인이 자연스럽게 발생한다.

**연속 환경** (Step 4): `Box(-1, 1, shape=(28,))`

| 구성 | 차원 | 내용 |
|------|------|------|
| robot_pos + heading | 4 | 위치(x,y) + 이동 방향(cos,sin) |
| nav_flags | 4 | 4방향(N/S/E/W) 이동 가능 여부 |
| top-5 crops | 20 | 가장 가까운 미처리 작물 5개의 (dx,dy,revealed,state) |

### 2.3 행동 공간 (Action Space)

**이산 환경**: `Discrete(7)` + **Action Masking**

| 인덱스 | 액션 | 마스킹 조건 |
|--------|------|------------|
| 0~3 | 이동 (상/하/좌/우) | 목적지가 작물/벽 |
| 4 | 예찰 (Scout) | 인접 미예찰 셀 없음 |
| 5 | 수확 (Harvest) | 인접 수확대기 셀 없음 |
| 6 | 방제 (Pest Control) | 인접 방제대기 셀 없음 |

> **Action Masking 효과**: 소프트맥스 전 로짓에 −∞ 마스크 → 무효 액션 원천 차단 → 초기 탐색 효율 대폭 향상

**연속 환경**: `Box(-1, 1, shape=(2,))` — 속도 벡터 (vx, vy)

### 2.4 보상 함수 (Reward Function)

**이산 환경 (Step 1 기준)**:

| 이벤트 | 보상 | 설계 근거 |
|--------|------|----------|
| 매 스텝 | −0.1 | 불필요한 배회 억제 |
| 충돌 시도 | −2.0 | 안전 항법 강제 |
| 신규 셀 예찰 | +1.0 | 탐색 유인 |
| 정상 확인 | +0.5 | 예찰 추가 인센티브 |
| 수확 성공 | +10.0 | 핵심 목표 |
| 방제 성공 | +8.0 | 핵심 목표 |
| 전체 완료 | +20.0 | 커버리지 완성 유도 |

> **Step 2/3 계층형 추가**: 레인 완료 시 +10.0 보너스(`REWARD_LANE_COMPLETE`) 추가.

**연속 환경 (Step 4)**: 스케일 조정 (R_STEP=−0.05, R_COLLISION=−1.0, R_SCOUT_NEW=+1.5) + Potential-based shaping `F = γ·φ(s') − φ(s)`, φ(s) = −min_dist × 0.3  
(단순 근접 보상은 시작점 정착 함정 유발 → 잠재 함수 차분 방식으로 국소 최적 방지)

### 2.5 에피소드 종료 조건

```python
# 성공 종료 (terminated) — Step 1 기준
done = all(crop_states[c] in {1, 4, 5} for c in all_crop_cells)
# 모든 C 셀이 정상확인(1) OR 수확완료(4) OR 방제완료(5)

# 시간 초과 (truncated)
done = step_count >= max_steps
```

| 모델 | 성공 종료 기준 | max_steps |
|------|--------------|-----------|
| Step 1 FarmEnv | 전체 작물 완료 | 540 |
| Step 2/3 LaneExecutorEnv | **목표 레인** 작물 완료 | **180** (레인당) |
| Step 4 연속 환경 | 전체 작물 완료 | 1200 |

### 2.6 작물 상태 초기 분포

| 상태 | 확률 | 의미 |
|------|------|------|
| 정상 (1) | 60% | 조치 불필요 |
| 수확 필요 (2) | 25% | 성숙 과일 수확 |
| 방제 필요 (3) | 15% | 해충 방제 |

---

## 3. 알고리즘 선택 및 적용

처음엔 단순하게 Flat RL로 시작했다가, 한계가 보일 때마다 계층 구조 → 거리 인식 → 연속 환경 순으로 단계적으로 발전시켰다. 각 단계에서 왜 바꿨는지 설명한다.

### 3.1 Step 1: MaskablePPO (이산 Flat RL)

**선택 이유**: 이산 행동 공간 + 동적 Action Masking 필요 → sb3-contrib의 `MaskablePPO`가 유일하게 두 조건 모두 지원.

| 파라미터 | 값 |
|----------|----|
| n_envs | 16 (DummyVecEnv) |
| n_steps | 512 |
| batch_size | 256 |
| learning_rate | 3×10⁻⁴ |
| ent_coef | 0.01 |
| total_timesteps | 1,500,000 |

### 3.2 Step 2: Hierarchical RL (계층적 강화학습)

**선택 이유**: Flat RL은 "어느 레인 방문" (전역 계획)과 "레인 내 행동" (국소 실행)이 혼재 → 계층 분리로 각 문제에 특화된 정책 학습.

```
High-level (PPO):  레인 선택 (Discrete 5) — obs: 레인별 완료율 + 거리 (10-dim)
       ↓ 목표 레인 지정
Low-level (MaskablePPO):  레인 처리 (Discrete 7 + masking) — obs: 전체 맵 + ch4 (목표 레인)
```

### 3.3 Step 3: Goal-reaching + DQN High-level

**선택 이유**: High-level obs에 거리 정보 추가 → 가장 가까운 레인 선택 학습 가능.  
DQN (off-policy): 느린 HL env에서 sample-efficient 학습.

### 3.4 Step 4: SAC (연속 환경)

**선택 이유**: 연속 2D 이동 제어 → continuous action space 필수 → SAC 선택.  
**최대 엔트로피 프레임워크**: 희박 보상 환경에서 자동 탐색 조절 (TD3/TQC 대비 유리).

---

## 4. 실험 과정

### 4.1 코드 구조

```
rlproject/
├── env/
│   ├── constants.py              # 상수 (셀 타입, 보상, 액션)
│   ├── map_generator.py          # 2열 재배단 맵 생성
│   ├── farm_env.py               # Step 1 이산 Gymnasium 환경
│   ├── continuous_farm_env.py    # Step 4 연속 환경
│   ├── continuous_farm_env_curriculum.py  # 커리큘럼 + nav flags
│   └── hierarchical/
│       ├── lane_executor_env.py  # Step 2/3 Low-level 환경
│       └── high_level_env.py     # Step 2/3 High-level 환경
├── train.py                      # Step 1 학습
├── train_hierarchical.py         # Step 2 학습
├── train_step3.py                # Step 3 학습
├── train_sac_curriculum.py       # Step 4 SAC 학습
├── train_recurrent_ppo.py        # RecurrentPPO 실험
├── train_td3.py                  # TD3 실험
├── train_tqc.py                  # TQC 실험
├── evaluate*.py                  # 평가 스크립트
├── record_gif*.py                # 데모 영상 생성
└── models/                       # 학습된 모델
```

### 4.2 실행 방법

```bash
# 환경 설치
pip install -r requirements.txt

# Step 1 학습
python train.py

# Step 2/3 학습
python train_hierarchical.py
python train_step3.py

# Step 4 SAC 학습
python train_sac_curriculum.py

# 평가
python evaluate.py                  # Step 1 vs Step 2
python evaluate_step3.py            # Step 1 vs 2 vs 3
python evaluate_greedy_lane.py      # Greedy vs RL 비교

# 영상 생성
python record_gif.py                # Step 1
python record_gif_step3.py          # Step 3
python record_gif_sac_curriculum.py # Step 4 SAC
```

### 4.3 환경 개선 이력

| 버전 | 변경 사항 | 이유 |
|------|----------|------|
| v1 | 1열 재배단 (24셀) | 초기 설계 |
| **v2** | **2열 재배단 (64셀)** | 현실 농경지: 안쪽 열만 스캔 가능 |
| v3 (연속) | 초기 obs=124-dim | 모든 작물 포함 |
| **v4 (연속)** | **obs=28-dim + nav_flags** | 희박 보상 문제 해결 |

### 4.4 학습 과정 요약

| 단계 | 알고리즘 | 학습 스텝 | GPU | 주요 기술 |
|------|----------|----------|-----|----------|
| Step 1 | MaskablePPO | 1.5M | RTX 3080 | Action Masking, 16 parallel envs |
| Step 2 LL | MaskablePPO | 500k | RTX 3080 | RandomLaneWrapper |
| Step 2 HL | PPO | 50k | RTX 3080 | distance-aware obs (10-dim) |
| Step 3 LL | MaskablePPO | 700k | RTX 3080 | Goal-reaching reward (+2.0) |
| Step 3 HL | DQN | 30k | RTX 3080 | Off-policy, nearest-lane learning |
| Step 4 | SAC | 5M | RTX 3080 | curriculum, nav_flags, potential shaping |

---

## 5. 결과 분석

### 5.1 Step별 최종 성능 비교

| 단계 | 알고리즘 | 성공률 | 커버리지 | 평균 스텝 | 핵심 기여 |
|------|----------|--------|---------|---------|-----------|
| Step 1 | MaskablePPO | 86~96% | 99.8% | 147~211 | Action Masking, Boustrophedon 자발 학습 |
| Step 2 | Hierarchical PPO | **98%** | **99.8%** | 198 | 계층 분리, 암묵적 다중 레인 커버리지 |
| **Step 3** | **Goal+DQN** | **100%** | **100%** | **141 ± 3** | 거리 인식 obs, Greedy 최적 정책 학습 |
| Step 4 | SAC+커리큘럼 | 96.7% | 99.9% | 157 | 연속 공간, nav_flags, 최대 엔트로피 |

*(results_step3_comparison.png, results_evaluation.png 참조)*

### 5.2 주요 발견 및 분석

**1. Action Masking이 생각보다 훨씬 중요했다**  
처음에는 충돌 패널티(-2.0)만으로도 에이전트가 알아서 피할 거라 생각했는데, 실제로 RecurrentPPO로 마스킹 없이 돌려보니 0.9% 커버리지로 완전히 실패했다. 마스킹이 없으면 초기 탐색에서 무효 액션에 샘플을 낭비해서 수렴 자체가 안 된다.

**2. Boustrophedon 패턴의 자발적 학습**  
명시적인 경로 알고리즘 없이 보상 설계(스텝 패널티 + 완료 보너스)만으로 지그재그(Boustrophedon) 탐색 패턴을 스스로 학습. 보상 기반 정책 발견 능력 실증.

**3. 보상 건드리는 것보다 관측 설계가 훨씬 효과적이었다**  
패널티를 −0.1에서 −0.3으로 올렸더니 오히려 성공률이 96%→80%로 떨어졌다. 반면 High-level obs에 거리 정보 하나 추가했더니 90%→100%로 올랐다. 결국 에이전트한테 필요한 정보를 제대로 주는 게 보상 튜닝보다 훨씬 중요하다는 걸 이번 실험에서 직접 확인했다.

**4. 좋은 obs를 주면 RL이 알아서 최적 전략을 찾는다**  
거리 정보를 obs에 넣고 나서 DQN이 학습한 정책을 분석해보니, Greedy nearest-lane(가장 가까운 레인 선택)이랑 결과가 완전히 똑같았다 (141±3 steps, 2.0 visits). 따로 "가까운 것부터 가라"고 가르치지 않았는데 거리 정보만 줬더니 스스로 그 전략을 발견한 것이다.

**5. 연속 환경에서는 SAC가 압도적이었다**  
같은 환경에서 TD3, TQC, SAC를 비교해봤는데:

| 알고리즘 | 성공률 | 커버리지 | 핵심 차이 |
|----------|--------|---------|-----------|
| TD3 | 0% | 42.3% | 결정론적 정책, 탐색 부족 |
| TQC | 3.3% | 60.0% | Q값 안정적이나 엔트로피 없음 |
| **SAC** | **96.7%** | **99.9%** | 최대 엔트로피 → 자동 탐색 조절 |

→ 희박 보상 + 넓은 탐색 공간에서 SAC의 엔트로피 목적함수가 결정적으로 유리.

*(results_algo_comparison.png 참조)*

**6. 구현/학습 실패 사례 및 해결**  

| 실패 | 원인 | 해결 방법 |
|------|------|----------|
| SAC 초기 커버리지 0% | 희박 보상 + 124-dim obs | obs 28-dim 단순화 + nav_flags 추가 |
| SAC 진동 정착 함정 | 단순 근접 보상 | Potential-based shaping으로 교체 |
| Step 패널티 −0.3 | 레인 완료 실패율 증가 | −0.1 유지 (보상 스케일 과잉 방지) |
| HL 보상 재방문 함정 | 완료 레인 재방문 시 보너스 | was_already_done 체크 추가 |

---

## 6. 종합 결론

### 6.1 달성 목표

| 항목 | 목표 | 달성 |
|------|------|------|
| 커버리지 완성 | 100% | Step 3: **100%** ✅ |
| 충돌 없는 이동 | 0 충돌 | 모든 Step: **0 충돌** ✅ |
| 상태 기반 조치 | 정확 분류 | Action Masking으로 **오분류 없음** ✅ |
| 연속 공간 적용 | SAC 타당성 | **96.7% 성공** ✅ |

### 6.2 실용적 함의

실제 로봇에 적용한다면 이런 순서가 현실적이다:

1. **시뮬레이션**: MaskablePPO(이산) 또는 SAC+curriculum(연속)으로 빠른 pre-training
2. **실제 로봇 배포**: SAC fine-tuning (off-policy → 실제 env에서 샘플 효율 우위)
3. **ROS2 연동**: 학습된 정책을 RLAgentNode로 배포, 나머지 노드와 토픽 통신

### 6.3 향후 연구

- **Multi-Agent RL (MARL)**: MAPPO 기반 복수 로봇 분할 예찰로 처리 시간 단축
- **실제 ROS2 배포**: Gymnasium 환경을 실제 ROS2 노드로 분리
- **동적 환경**: 작물 성장 주기, 이동 장애물 등 실제 요소 반영
- **SAC-Discrete**: 이산 환경에서도 최대 엔트로피 RL 적용 (커스텀 구현 필요)

---

## 영상 클립 목록

| 파일 | 내용 |
|------|------|
| `agent_demo.gif` | Step 1 — Flat MaskablePPO |
| `agent_demo_hierarchical.gif` | Step 2 — Hierarchical PPO |
| `agent_demo_step3.gif` | Step 3 — Goal+DQN (100% success) |
| `agent_demo_greedy.gif` | 비교 — Greedy nearest-lane |
| `agent_demo_sac_curriculum.gif` | Step 4 — SAC Curriculum (96.7%) |
| `agent_demo_td3.gif` | 비교 — TD3 (42% coverage) |
| `agent_demo_tqc.gif` | 비교 — TQC (60% coverage) |

---

## 참고 문헌

- Gymnasium: Farama Foundation (2023). https://gymnasium.farama.org/
- Stable-Baselines3: Raffin et al. (2021). JMLR.
- sb3-contrib: https://sb3-contrib.readthedocs.io/
- MaskablePPO: Huang & Ontañón (2022). A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.
- Boustrophedon Coverage: Choset, H. (2001). Coverage for robotics — A survey of recent results. Ann. Math. Artif. Intell.
- Hierarchical RL: Nachum, O. et al. (2018). Data-Efficient Hierarchical Reinforcement Learning (HIRO). NeurIPS.
- Option-Critic: Bacon, P.-L. et al. (2017). The Option-Critic Architecture. AAAI.
- SAC: Haarnoja et al. (2018). Soft Actor-Critic. ICML.
- TD3: Fujimoto et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods. ICML.
- TQC: Kuznetsov et al. (2020). Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics. ICML.
- Potential-based Reward Shaping: Ng, Russell & Bartlett (1999). Policy Invariance Under Reward Transformations. ICML.
