import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import io
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env.continuous_farm_env_curriculum import ContinuousFarmEnvCurriculum
from env.continuous_farm_env import (
    CELL_SIZE, STATE_UNKNOWN, STATE_NORMAL_DONE,
    STATE_HARVEST_PENDING, STATE_PEST_PENDING,
    STATE_HARVEST_DONE, STATE_PEST_DONE, SCOUT_RADIUS,
)

STATE_COLORS = {
    STATE_UNKNOWN: '#D8D8D8', STATE_NORMAL_DONE: '#33C733',
    STATE_HARVEST_PENDING: '#FFD700', STATE_PEST_PENDING: '#FF5A1A',
    STATE_HARVEST_DONE: '#007A00', STATE_PEST_DONE: '#8B2500',
}
LEGEND_PATCHES = [
    patches.Patch(color='#D8D8D8', label='Unknown'),
    patches.Patch(color='#33C733', label='Normal'),
    patches.Patch(color='#FFD700', label='Harvest Pending'),
    patches.Patch(color='#FF5A1A', label='Pest Pending'),
    patches.Patch(color='#007A00', label='Harvest Done'),
    patches.Patch(color='#8B2500', label='Pest Done'),
]


def render_frame(env, step, coverage, trajectory):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#222222')
    G_H, G_W = env.G_H, env.G_W

    for row in [1, G_H - 2]:
        ax.add_patch(patches.Rectangle((CELL_SIZE, row * CELL_SIZE),
            (G_W - 2) * CELL_SIZE, CELL_SIZE, lw=0, fc='#F0F0F0', alpha=0.9))
    for row in range(2, G_H - 2):
        for col in range(1, G_W - 1):
            if (col - 1) % 3 == 0:
                ax.add_patch(patches.Rectangle((col * CELL_SIZE, row * CELL_SIZE),
                    CELL_SIZE, CELL_SIZE, lw=0, fc='#F0F0F0', alpha=0.9))

    for i, (cx, cy) in enumerate(env.crop_centres):
        ax.add_patch(patches.Rectangle(
            (cx - 0.5*CELL_SIZE, cy - 0.5*CELL_SIZE), CELL_SIZE, CELL_SIZE,
            lw=0.3, ec='#555', fc=STATE_COLORS[env.crop_states[i]]))

    rx, ry = env.robot_pos
    ax.add_patch(plt.Circle((rx, ry), SCOUT_RADIUS,
        color='cyan', fill=False, lw=1.0, ls='--', alpha=0.4))

    if len(trajectory) > 1:
        traj = np.array(trajectory[-100:])
        ax.plot(traj[:,0], traj[:,1], color='dodgerblue', lw=0.8, alpha=0.5)
    ax.add_patch(plt.Circle((rx, ry), 0.25, color='royalblue', zorder=10,
                             lw=1.5, ec='white'))

    ax.set_xlim(0, G_W*CELL_SIZE); ax.set_ylim(0, G_H*CELL_SIZE)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'[Step4 SAC + Simplified Obs] Step {step:4d}  |  Coverage {coverage:.0%}',
                 fontsize=10, pad=6)
    ax.legend(handles=LEGEND_PATCHES, loc='upper right',
              bbox_to_anchor=(1.35, 1.02), fontsize=7, framealpha=0.9)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    return Image.open(buf).copy()


def record(seed=3, out='assets/demos/step4.gif', fps=8):
    def make_env():
        env = ContinuousFarmEnvCurriculum(); env.curriculum_level = 2
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load('models/sac_curriculum_vecnorm.pkl', vec_env)
    vec_env.training = False; vec_env.norm_reward = False
    model = SAC.load('models/sac_curriculum', env=vec_env)
    # VecNormalize wraps DummyVecEnv; access raw env via .venv.envs[0].unwrapped
    raw = vec_env.venv.envs[0].unwrapped

    obs = vec_env.reset()
    trajectory = [raw.robot_pos.copy()]
    frames = [render_frame(raw, 0, raw._coverage_rate(), trajectory)]

    steps, done = 0, False
    final_coverage = 0.0
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = vec_env.step(a)
        # Record BEFORE auto-reset overwrites raw state
        curr_cov = info[0].get('coverage', raw._coverage_rate())
        trajectory.append(raw.robot_pos.copy()); steps += 1
        if done[0]:
            final_coverage = curr_cov   # save before env resets
        frames.append(render_frame(raw, steps, curr_cov, trajectory))

    frames += [frames[-1]] * (fps * 2)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000/fps), loop=0)
    print(f'Saved: {out}  ({len(frames)} frames, {steps} steps, '
          f'coverage={final_coverage:.0%})')
    vec_env.close()


if __name__ == '__main__':
    record(seed=3, out='assets/demos/step4.gif', fps=8)
