# 정밀농업 자율 로봇 강화학습

부분 관측 온실에서 자율 로봇이 작물 상태를 확인하고 필요한 작업을 수행하도록 학습하는 커스텀 강화학습 프로젝트입니다. 단일 이산 정책에서 계층형 정책과 연속 제어까지 네 단계로 확장하며 환경·관측·알고리즘 설계가 성능에 미치는 영향을 비교합니다.

## 핵심 결과

| 단계 | 환경 및 정책 | 성공률 | 평균 스텝 |
|---|---|---:|---:|
| Step 1 | MaskablePPO 기반 Flat RL | 96% | 147 ± 80 |
| Step 2 | Hierarchical PPO | 98% | 198 ± 349 |
| Step 3 | 거리 관측 + DQN High-level | **100%** | **141 ± 3** |
| Step 4 | SAC + 단순화 관측 기반 연속 제어 | 96.7% | 157 ± 194 |

Step 3의 거리 관측·DQN·학습 설정을 포함한 최종 구성은 가장 가까운 미완료 레인을 선택하는 안정적인 정책을 학습했습니다. 연속 제어에서도 `nav_flags`와 관측 단순화를 포함한 최종 구성이 가장 높은 성능을 기록했습니다. 각 변경의 독립적인 인과 효과는 추가 통제 실험이 필요합니다.

자세한 환경 설계, 실험 결과와 시행착오는 [연구 보고서](docs/report.md)에서 확인할 수 있습니다. 제출용 문서는 [최종 PDF](docs/RLproject_report_final.pdf)로 보존했습니다.

## 단계별 데모

| Step 1: Flat RL | Step 2: Hierarchical PPO |
|---|---|
| ![Step 1 demo](assets/demos/step1.gif) | ![Step 2 demo](assets/demos/step2.gif) |

| Step 3: Distance + DQN | Step 4: Continuous SAC |
|---|---|
| ![Step 3 demo](assets/demos/step3.gif) | ![Step 4 demo](assets/demos/step4.gif) |

## 저장소 구조

```text
.
├── env/                  # Gymnasium 환경 구현
│   ├── farm_env.py       # Step 1 이산 환경
│   ├── hierarchical/     # Step 2/3 계층형 환경
│   └── continuous_*.py   # Step 4 연속 환경
├── scripts/
│   ├── train/            # 대표 Step 1~4 학습 진입점
│   ├── evaluate/         # 성능 및 Greedy 비교
│   ├── record/           # Step 3/4 GIF 생성
│   └── experiments/      # 비교·보조·실패 실험
├── tests/                # 환경 동작 테스트
├── models/               # 학습 모델 저장 위치(Git 제외)
├── assets/
│   ├── demos/            # Step 1~4 GIF
│   └── figures/          # 보고서 결과 그림
└── docs/                 # Markdown 보고서 및 최종 PDF
```

## 설치

Python 3.10 환경을 권장합니다.

```bash
pip install -r requirements.txt
```

## 대표 실험 실행

모든 명령은 저장소 루트에서 실행합니다. 학습된 모델은 `models/`에 저장되며 Git에는 포함하지 않습니다.

```bash
# Step 1~4 학습
python -m scripts.train.step1
python -m scripts.train.step2
python -m scripts.train.step3
python -m scripts.train.step4

# 평가
python -m scripts.evaluate.compare_steps
python -m scripts.evaluate.compare_greedy

# 데모 생성
python -m scripts.record.step3
python -m scripts.record.step4
```

추가 알고리즘 비교 및 실패 실험은 `scripts/experiments/`에 보존되어 있습니다.

## 테스트

```bash
pytest -q
```

## 재현 범위

학습 모델 바이너리와 원시 로그는 저장소 크기를 줄이기 위해 포함하지 않습니다. 평가 수치를 독립적으로 재현하려면 먼저 대응하는 학습 스크립트를 실행해야 하며, 개발 초기 일부 실험은 현재 코드만으로 완전히 재현되지 않습니다.

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX 3080 (CUDA 12.4) |
| PyTorch | 2.5.1+cu124 |
| Python | 3.10 |
| 평가 에피소드 | 각 단계 ≥ 20 |
