import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import io
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN
from env.hierarchical.high_level_env import HighLevelFarmEnv
from env.constants import (
    CELL_PATH, CELL_CROP, CELL_WALL,
    STATE_UNKNOWN, STATE_NORMAL_DONE,
    STATE_HARVEST_PENDING, STATE_PEST_PENDING,
    STATE_HARVEST_DONE, STATE_PEST_DONE,
)

STATE_COLORS = {
    STATE_UNKNOWN:         [0.85, 0.85, 0.85],
    STATE_NORMAL_DONE:     [0.20, 0.78, 0.20],
    STATE_HARVEST_PENDING: [1.00, 0.85, 0.00],
    STATE_PEST_PENDING:    [1.00, 0.35, 0.10],
    STATE_HARVEST_DONE:    [0.00, 0.45, 0.00],
    STATE_PEST_DONE:       [0.55, 0.15, 0.00],
}
CELL_COLORS = {CELL_PATH: [0.97, 0.97, 0.97], CELL_WALL: [0.15, 0.15, 0.15]}
TARGET_TINT = [0.55, 0.85, 1.00]

LEGEND_ITEMS = [
    mpatches.Patch(color=STATE_COLORS[STATE_UNKNOWN],         label='Unknown'),
    mpatches.Patch(color=STATE_COLORS[STATE_NORMAL_DONE],     label='Normal'),
    mpatches.Patch(color=STATE_COLORS[STATE_HARVEST_PENDING], label='Harvest Pending'),
    mpatches.Patch(color=STATE_COLORS[STATE_PEST_PENDING],    label='Pest Pending'),
    mpatches.Patch(color=STATE_COLORS[STATE_HARVEST_DONE],    label='Harvest Done'),
    mpatches.Patch(color=STATE_COLORS[STATE_PEST_DONE],       label='Pest Done'),
    mpatches.Patch(color=[0.10, 0.10, 0.90],                  label='Robot'),
    mpatches.Patch(color=TARGET_TINT,                          label='Target Lane'),
]


def render_frame(inner_env, step, coverage, lane_visit, target_col):
    fig, ax = plt.subplots(figsize=(7, 5))
    H, W = inner_env.H, inner_env.W
    img = np.ones((H, W, 3))
    for r in range(H):
        for c in range(W):
            if inner_env.layout[r, c] == CELL_CROP:
                img[r, c] = STATE_COLORS[inner_env.crop_states[r, c]]
            else:
                img[r, c] = CELL_COLORS[inner_env.layout[r, c]]
    for r in range(H):
        if inner_env.layout[r, target_col] == CELL_PATH:
            img[r, target_col] = TARGET_TINT
    ra, ca = inner_env.agent_pos
    img[ra, ca] = [0.10, 0.10, 0.90]

    ax.imshow(img, interpolation='nearest', aspect='equal')
    ax.set_title(
        f'[Step3 DQN] Step {step:3d}  |  Coverage {coverage:.0%}  |  Lane Visit #{lane_visit}  |  Target Col {target_col}',
        fontsize=9, pad=6,
    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=LEGEND_ITEMS, loc='upper right',
              bbox_to_anchor=(1.42, 1.02), fontsize=7, framealpha=0.9)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def record(seed=5, out='assets/demos/step3.gif', fps=5):
    ll_model = MaskablePPO.load('models/lane_executor_s3')
    hl_model = DQN.load('models/high_level_s3')

    hl_env = HighLevelFarmEnv(ll_model, n_beds=4, field_height=8)
    obs, _ = hl_env.reset(seed=seed)
    inner = hl_env.inner
    frames = [render_frame(inner, 0, inner._coverage_rate(), 0, inner.lane_cols[0])]

    total_steps = 0
    lane_visit = 0
    terminated = truncated = False

    while not (terminated or truncated):
        hl_action, _ = hl_model.predict(obs, deterministic=True)
        target_col = hl_env.lane_cols[int(hl_action)]
        lane_visit += 1

        inner.target_lane_col = target_col
        inner.step_count = 0
        inner._goal_reached = False
        ll_obs = inner._get_obs()

        lane_done = lane_trunc = False
        while not (lane_done or lane_trunc):
            ll_action, _ = ll_model.predict(
                ll_obs, deterministic=True, action_masks=inner.action_masks()
            )
            ll_obs, _, lane_done, lane_trunc, info = inner.step(int(ll_action))
            total_steps += 1
            frames.append(render_frame(inner, total_steps, info['coverage'], lane_visit, target_col))

        obs = hl_env._get_hl_obs()
        terminated = inner._is_complete()
        truncated = (hl_env._lane_visits >= hl_env.max_lane_visits) and not terminated
        hl_env._lane_visits = lane_visit

    frames += [frames[-1]] * (fps * 2)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    print(f'Saved: {out}  ({len(frames)} frames, {total_steps} steps, {lane_visit} lane visits)')
    print(f'Final: coverage={info["coverage"]:.0%}')


if __name__ == '__main__':
    record(seed=5, out='assets/demos/step3.gif', fps=5)
