import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image
import io
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env.continuous_farm_env import (
    ContinuousFarmEnv, CELL_SIZE,
    STATE_UNKNOWN, STATE_NORMAL_DONE,
    STATE_HARVEST_PENDING, STATE_PEST_PENDING,
    STATE_HARVEST_DONE, STATE_PEST_DONE,
    SCOUT_RADIUS, ACT_RADIUS, DONE_STATES,
)

STATE_COLORS = {
    STATE_UNKNOWN:         '#D8D8D8',
    STATE_NORMAL_DONE:     '#33C733',
    STATE_HARVEST_PENDING: '#FFD700',
    STATE_PEST_PENDING:    '#FF5A1A',
    STATE_HARVEST_DONE:    '#007A00',
    STATE_PEST_DONE:       '#8B2500',
}

LEGEND_PATCHES = [
    patches.Patch(color=STATE_COLORS[STATE_UNKNOWN],         label='Unknown'),
    patches.Patch(color=STATE_COLORS[STATE_NORMAL_DONE],     label='Normal'),
    patches.Patch(color=STATE_COLORS[STATE_HARVEST_PENDING], label='Harvest Pending'),
    patches.Patch(color=STATE_COLORS[STATE_PEST_PENDING],    label='Pest Pending'),
    patches.Patch(color=STATE_COLORS[STATE_HARVEST_DONE],    label='Harvest Done'),
    patches.Patch(color=STATE_COLORS[STATE_PEST_DONE],       label='Pest Done'),
]


def render_frame(env, step, coverage, trajectory):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Background: wall (dark)
    ax.set_facecolor('#222222')

    # Draw passable areas (headlands + lanes) in light grey
    G_H, G_W = env.G_H, env.G_W
    # headlands
    for row in [1, G_H - 2]:
        rect = patches.Rectangle(
            (CELL_SIZE, row * CELL_SIZE),
            (G_W - 2) * CELL_SIZE, CELL_SIZE,
            linewidth=0, facecolor='#F0F0F0', alpha=0.9
        )
        ax.add_patch(rect)
    # driving lanes (field rows)
    for row in range(2, G_H - 2):
        for col in range(1, G_W - 1):
            if (col - 1) % 3 == 0:   # lane col
                rect = patches.Rectangle(
                    (col * CELL_SIZE, row * CELL_SIZE),
                    CELL_SIZE, CELL_SIZE,
                    linewidth=0, facecolor='#F0F0F0', alpha=0.9
                )
                ax.add_patch(rect)

    # Draw crop cells
    for i, (cx, cy) in enumerate(env.crop_centres):
        col_left = cx - 0.5 * CELL_SIZE
        row_bottom = cy - 0.5 * CELL_SIZE
        color = STATE_COLORS[env.crop_states[i]]
        rect = patches.Rectangle(
            (col_left, row_bottom), CELL_SIZE, CELL_SIZE,
            linewidth=0.3, edgecolor='#555', facecolor=color
        )
        ax.add_patch(rect)

    # Draw scout radius circle (dashed)
    rx, ry = env.robot_pos
    scout_circle = plt.Circle((rx, ry), SCOUT_RADIUS,
                               color='cyan', fill=False, linewidth=1.0,
                               linestyle='--', alpha=0.5)
    act_circle = plt.Circle((rx, ry), ACT_RADIUS,
                             color='yellow', fill=False, linewidth=1.0,
                             linestyle=':', alpha=0.5)
    ax.add_patch(scout_circle)
    ax.add_patch(act_circle)

    # Trajectory (past positions)
    if len(trajectory) > 1:
        traj = np.array(trajectory[-80:])
        ax.plot(traj[:, 0], traj[:, 1], color='dodgerblue',
                linewidth=0.8, alpha=0.5)

    # Robot
    robot_dot = plt.Circle((rx, ry), 0.25, color='royalblue',
                            zorder=10, linewidth=1.5, edgecolor='white')
    ax.add_patch(robot_dot)

    ax.set_xlim(0, G_W * CELL_SIZE)
    ax.set_ylim(0, G_H * CELL_SIZE)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f'[Step4 SAC Continuous] Step {step:4d}  |  Coverage {coverage:.0%}',
        fontsize=10, pad=6
    )
    ax.legend(handles=LEGEND_PATCHES, loc='upper right',
              bbox_to_anchor=(1.35, 1.02), fontsize=7, framealpha=0.9)

    # Labels for circles
    ax.text(rx + SCOUT_RADIUS + 0.05, ry, 'Scout r', fontsize=6,
            color='cyan', alpha=0.7, va='center')

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def record(seed=7, out='agent_demo_sac.gif', fps=8, max_steps=400):
    def make_env_fn():
        return Monitor(ContinuousFarmEnv(n_beds=3, field_height=5))

    vec_env = DummyVecEnv([make_env_fn])
    vec_env = VecNormalize.load('models/sac_vecnormalize.pkl', vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = SAC.load('models/sac_continuous', env=vec_env)

    # Access the raw env
    raw_env = vec_env.envs[0].unwrapped
    obs = vec_env.reset()
    # Fix seed for raw env
    raw_env._rng = np.random.default_rng(seed)

    trajectory = [raw_env.robot_pos.copy()]
    frames = [render_frame(raw_env, 0, raw_env._coverage_rate(), trajectory)]

    steps = 0
    done = False
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = vec_env.step(action)
        trajectory.append(raw_env.robot_pos.copy())
        steps += 1
        frames.append(render_frame(raw_env, steps,
                                   raw_env._coverage_rate(), trajectory))

    # Hold final frame
    frames += [frames[-1]] * (fps * 2)

    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f'Saved: {out}  ({len(frames)} frames, {steps} steps)')
    print(f'Final coverage: {raw_env._coverage_rate():.0%}')
    vec_env.close()


if __name__ == '__main__':
    record(seed=7, out='agent_demo_sac.gif', fps=8, max_steps=400)
