import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import io
from sb3_contrib import MaskablePPO
from env.farm_env import FarmEnv
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

LEGEND_ITEMS = [
    mpatches.Patch(color=STATE_COLORS[STATE_UNKNOWN],         label='Unknown'),
    mpatches.Patch(color=STATE_COLORS[STATE_NORMAL_DONE],     label='Normal'),
    mpatches.Patch(color=STATE_COLORS[STATE_HARVEST_PENDING], label='Harvest Pending'),
    mpatches.Patch(color=STATE_COLORS[STATE_PEST_PENDING],    label='Pest Pending'),
    mpatches.Patch(color=STATE_COLORS[STATE_HARVEST_DONE],    label='Harvest Done'),
    mpatches.Patch(color=STATE_COLORS[STATE_PEST_DONE],       label='Pest Done'),
    mpatches.Patch(color=[0.10, 0.10, 0.90],                  label='Robot'),
]


def render_frame(env, step, reward, coverage):
    fig, ax = plt.subplots(figsize=(6, 5))
    img = np.ones((env.H, env.W, 3))
    for r in range(env.H):
        for c in range(env.W):
            if env.layout[r, c] == CELL_CROP:
                img[r, c] = STATE_COLORS[env.crop_states[r, c]]
            else:
                img[r, c] = CELL_COLORS[env.layout[r, c]]
    ra, ca = env.agent_pos
    img[ra, ca] = [0.10, 0.10, 0.90]

    ax.imshow(img, interpolation='nearest', aspect='equal')
    ax.set_title(f'Step {step:3d}  |  Coverage {coverage:.0%}  |  Cumulative Reward {reward:+.1f}',
                 fontsize=10, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=LEGEND_ITEMS, loc='upper right',
              bbox_to_anchor=(1.38, 1.02), fontsize=7, framealpha=0.9)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def record(seed=3, out='agent_demo.gif', fps=6):
    env = FarmEnv(n_beds=4, field_height=8)
    model = MaskablePPO.load('models/farm_ppo')

    obs, _ = env.reset(seed=seed)
    frames = [render_frame(env, 0, 0.0, env._coverage_rate())]

    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        frames.append(render_frame(env, env.step_count, total_reward, info['coverage']))

    # Hold last frame longer (x4)
    frames += [frames[-1]] * (fps * 2)

    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f'Saved: {out}  ({len(frames)} frames, {env.step_count} steps)')
    print(f'Final: coverage={info["coverage"]:.0%}, reward={total_reward:.1f}')


if __name__ == '__main__':
    record(seed=3, out='agent_demo.gif', fps=6)
