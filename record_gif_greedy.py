import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import io
from sb3_contrib import MaskablePPO
from env.hierarchical.high_level_env import HighLevelFarmEnv
from env.constants import (CELL_PATH, CELL_CROP, CELL_WALL, DONE_STATES,
    STATE_UNKNOWN, STATE_NORMAL_DONE, STATE_HARVEST_PENDING,
    STATE_PEST_PENDING, STATE_HARVEST_DONE, STATE_PEST_DONE)

STATE_COLORS = {
    STATE_UNKNOWN:'#D8D8D8', STATE_NORMAL_DONE:'#33C733',
    STATE_HARVEST_PENDING:'#FFD700', STATE_PEST_PENDING:'#FF5A1A',
    STATE_HARVEST_DONE:'#007A00', STATE_PEST_DONE:'#8B2500',
}
CELL_COLORS = {CELL_PATH:'#F0F0F0', CELL_WALL:'#222222'}
TARGET_TINT = [0.55, 0.85, 1.00]

LEGEND = [
    patches.Patch(color='#D8D8D8', label='Unknown'),
    patches.Patch(color='#33C733', label='Normal'),
    patches.Patch(color='#FFD700', label='Harvest Pending'),
    patches.Patch(color='#FF5A1A', label='Pest Pending'),
    patches.Patch(color='#007A00', label='Harvest Done'),
    patches.Patch(color='#8B2500', label='Pest Done'),
    patches.Patch(color=TARGET_TINT, label='Target Lane'),
]


def greedy_select(hl_env):
    agent_col = hl_env.inner.agent_pos[1]
    unfinished = [l for l in hl_env.lane_cols if not hl_env._is_lane_already_done(l)]
    return min(unfinished, key=lambda col: abs(col - agent_col)) if unfinished else hl_env.lane_cols[0]


def render_frame(inner, step, coverage, lane_visit, target_col):
    from env.farm_env import FarmEnv
    H, W = inner.H, inner.W
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor('#222222')
    img = np.ones((H, W, 3))
    for r in range(H):
        for c in range(W):
            if inner.layout[r, c] == CELL_CROP:
                img[r, c] = [int(STATE_COLORS[inner.crop_states[r,c]][i:i+2], 16)/255
                             for i in (1,3,5)]
            elif inner.layout[r, c] == CELL_PATH:
                img[r, c] = [0.97, 0.97, 0.97]
            else:
                img[r, c] = [0.15, 0.15, 0.15]
    # target lane highlight
    for r in range(H):
        if inner.layout[r, target_col] == CELL_PATH:
            img[r, target_col] = TARGET_TINT
    ra, ca = inner.agent_pos
    img[ra, ca] = [0.1, 0.1, 0.9]
    ax.imshow(img, interpolation='nearest', aspect='equal')
    ax.set_title(f'[Greedy Nearest-Lane] Step {step:3d} | Coverage {coverage:.0%} | Visit #{lane_visit} | Target Col {target_col}',
                 fontsize=9, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=LEGEND, loc='upper right', bbox_to_anchor=(1.42,1.02), fontsize=7, framealpha=0.9)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    return Image.open(buf).copy()


def record(seed=3, out='agent_demo_greedy.gif', fps=5):
    ll = MaskablePPO.load('models/lane_executor_s3')
    hl_env = HighLevelFarmEnv(ll, n_beds=4, field_height=8)
    obs, _ = hl_env.reset(seed=seed)
    inner = hl_env.inner

    frames = [render_frame(inner, 0, inner._coverage_rate(), 0, hl_env.lane_cols[0])]
    total_steps, lane_visit = 0, 0
    terminated = truncated = False

    while not (terminated or truncated):
        target_col = greedy_select(hl_env)
        action = hl_env.lane_cols.index(target_col)
        lane_visit += 1

        inner.target_lane_col = target_col
        inner.step_count = 0
        ll_obs = inner._get_obs()
        lane_done = lane_trunc = False

        while not (lane_done or lane_trunc):
            ll_action, _ = ll.predict(ll_obs, deterministic=True, action_masks=inner.action_masks())
            ll_obs, _, lane_done, lane_trunc, info = inner.step(int(ll_action))
            total_steps += 1
            frames.append(render_frame(inner, total_steps, info['coverage'], lane_visit, target_col))

        obs = hl_env._get_hl_obs()
        terminated = inner._is_complete()
        truncated = (hl_env._lane_visits >= hl_env.max_lane_visits) and not terminated
        hl_env._lane_visits = lane_visit

    frames += [frames[-1]] * (fps * 2)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0)
    print(f'Saved: {out} ({len(frames)} frames, {total_steps} steps, {lane_visit} visits, coverage={info["coverage"]:.0%})')


if __name__ == '__main__':
    record(seed=3, out='agent_demo_greedy.gif', fps=5)
