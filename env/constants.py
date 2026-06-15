# 맵 셀 종류(ch0)
CELL_PATH = 0
CELL_CROP = 1
CELL_WALL = 2

# 작물 관측 및 작업 상태(ch3)
STATE_UNKNOWN         = 0
STATE_NORMAL_DONE     = 1
STATE_HARVEST_PENDING = 2
STATE_PEST_PENDING    = 3
STATE_HARVEST_DONE    = 4
STATE_PEST_DONE       = 5

DONE_STATES = {STATE_NORMAL_DONE, STATE_HARVEST_DONE, STATE_PEST_DONE}

# 이산 행동 인덱스
ACT_UP      = 0
ACT_DOWN    = 1
ACT_LEFT    = 2
ACT_RIGHT   = 3
ACT_SCOUT   = 4
ACT_HARVEST = 5
ACT_PEST    = 6
N_ACTIONS   = 7

MOVE_DELTA = {
    ACT_UP:    (-1,  0),
    ACT_DOWN:  ( 1,  0),
    ACT_LEFT:  ( 0, -1),
    ACT_RIGHT: ( 0,  1),
}

# 공통 보상 상수
REWARD_STEP            = -0.1
REWARD_COLLISION       = -2.0
REWARD_SCOUT_NEW       =  1.0
REWARD_NORMAL_CONFIRM  =  0.5
REWARD_HARVEST         = 10.0
REWARD_PEST            =  8.0
REWARD_COMPLETION      = 20.0

# 초기 작물 상태 확률: 정상 / 수확 필요 / 방제 필요
CROP_STATE_PROBS = [0.60, 0.25, 0.15]
CROP_STATE_VALUES = [STATE_NORMAL_DONE, STATE_HARVEST_PENDING, STATE_PEST_PENDING]

# 계층형 강화학습 보상(Step 2)
REWARD_LANE_COMPLETE = 10.0   # 하위 정책: 목표 레인의 모든 작업 완료 보너스
REWARD_LANE_STEP     = -0.1   # 하위 정책: 스텝 비용(-0.3은 학습을 불안정하게 만들었음)
REWARD_HL_LANE_DONE  =  5.0   # 상위 정책: 새로 선택한 레인 완료 보너스
REWARD_HL_ALL_DONE   = 20.0   # 상위 정책: 전체 작물 완료 보너스
HL_STEP_COST         =  0.01  # 상위 정책: 하위 정책이 사용한 스텝당 비용 계수

# Step 3: 목표 레인 도달 보상
REWARD_GOAL_REACH    =  2.0   # 하위 정책: 에피소드에서 목표 레인에 처음 도착할 때 지급
