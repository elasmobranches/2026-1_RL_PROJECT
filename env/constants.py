# Cell types (ch0)
CELL_PATH = 0
CELL_CROP = 1
CELL_WALL = 2

# Crop/action states (ch3)
STATE_UNKNOWN         = 0
STATE_NORMAL_DONE     = 1
STATE_HARVEST_PENDING = 2
STATE_PEST_PENDING    = 3
STATE_HARVEST_DONE    = 4
STATE_PEST_DONE       = 5

DONE_STATES = {STATE_NORMAL_DONE, STATE_HARVEST_DONE, STATE_PEST_DONE}

# Action indices
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

# Reward constants
REWARD_STEP            = -0.1
REWARD_COLLISION       = -2.0
REWARD_SCOUT_NEW       =  1.0
REWARD_NORMAL_CONFIRM  =  0.5
REWARD_HARVEST         = 10.0
REWARD_PEST            =  8.0
REWARD_COMPLETION      = 20.0

# Crop state probability (normal / harvest / pest)
CROP_STATE_PROBS = [0.60, 0.25, 0.15]
CROP_STATE_VALUES = [STATE_NORMAL_DONE, STATE_HARVEST_PENDING, STATE_PEST_PENDING]

# Hierarchical RL (Step 2)
REWARD_LANE_COMPLETE = 10.0   # low-level: bonus when target lane fully processed
REWARD_HL_LANE_DONE  =  5.0   # high-level: bonus when dispatched lane completes
REWARD_HL_ALL_DONE   = 20.0   # high-level: bonus when all crops done
HL_STEP_COST         =  0.01  # high-level: per-step cost coefficient (not a reward, hence no REWARD_ prefix)
