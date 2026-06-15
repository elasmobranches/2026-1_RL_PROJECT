"""Step 3 상위 정책과 가장 가까운 미완료 레인 규칙의 성능을 비교한다."""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN
from env.hierarchical.high_level_env import HighLevelFarmEnv
from env.constants import DONE_STATES


def greedy_select_lane(hl_env):
    """현재 로봇 위치에서 가장 가까운 미완료 레인을 선택한다."""
    agent_col = hl_env.inner.agent_pos[1]
    unfinished = [
        lane_col for lane_col in hl_env.lane_cols
        if not hl_env._is_lane_already_done(lane_col)
    ]
    if not unfinished:
        return hl_env.lane_cols[0]  # 모든 레인이 완료된 경우의 안전한 기본값
    return min(unfinished, key=lambda col: abs(col - agent_col))


def run_greedy(ll_model, seed):
    env = HighLevelFarmEnv(ll_model, n_beds=4, field_height=8)
    obs, _ = env.reset(seed=seed)
    total_steps, visits, done, truncated = 0, 0, False, False
    while not (done or truncated):
        target_col = greedy_select_lane(env)
        action = env.lane_cols.index(target_col)
        obs, r, done, truncated, info = env.step(action)
        total_steps += info['steps_for_lane']
        visits += 1
    return info['coverage'], total_steps, done, visits


def run_rl(hl_model, ll_model, seed):
    env = HighLevelFarmEnv(ll_model, n_beds=4, field_height=8)
    obs, _ = env.reset(seed=seed)
    total_steps, visits, done, truncated = 0, 0, False, False
    while not (done or truncated):
        action, _ = hl_model.predict(obs, deterministic=True)
        obs, r, done, truncated, info = env.step(int(action))
        total_steps += info['steps_for_lane']
        visits += 1
    return info['coverage'], total_steps, done, visits


def evaluate(n=30):
    ll = MaskablePPO.load('models/lane_executor_s3')
    hl = DQN.load('models/high_level_s3')

    greedy_r, rl_r = [], []
    for seed in range(n):
        cov_g, steps_g, succ_g, vis_g = run_greedy(ll, seed)
        cov_r, steps_r, succ_r, vis_r = run_rl(hl, ll, seed)
        greedy_r.append((cov_g, steps_g, succ_g, vis_g))
        rl_r.append((cov_r, steps_r, succ_r, vis_r))

    def summary(results, label):
        covs  = [r[0] for r in results]
        steps = [r[1] for r in results]
        succs = [r[2] for r in results]
        visits= [r[3] for r in results]
        print(f'\n[{label}]')
        print(f'  Success: {np.mean(succs):.1%}  Coverage: {np.mean(covs):.1%}')
        print(f'  Avg steps: {np.mean(steps):.0f} ± {np.std(steps):.0f}')
        print(f'  Avg lane visits: {np.mean(visits):.1f}')
        return covs, steps, succs

    print(f'=== Greedy Nearest-Lane vs RL High-Level ({n} episodes) ===')
    g_covs, g_steps, g_succs = summary(greedy_r, 'Greedy nearest-lane')
    r_covs, r_steps, r_succs = summary(rl_r,    'RL DQN High-level')

    # 두 방법의 스텝 수와 커버리지 분포를 겹쳐 그린다.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('Greedy Nearest-Lane vs RL High-Level', fontsize=12, fontweight='bold')
    for ax, data_g, data_r, xlabel, title in [
        (axes[0], g_steps, r_steps, 'Steps', 'Completion Steps'),
        (axes[1], g_covs,  r_covs,  'Coverage', 'Coverage Rate'),
    ]:
        ax.hist(data_g, bins=12, alpha=0.6, color='steelblue', label=f'Greedy μ={np.mean(data_g):.1f}')
        ax.hist(data_r, bins=12, alpha=0.6, color='coral',     label=f'RL DQN μ={np.mean(data_r):.1f}')
        ax.axvline(np.mean(data_g), color='blue', lw=2, ls='--')
        ax.axvline(np.mean(data_r), color='red',  lw=2, ls='--')
        ax.set_title(title); ax.set_xlabel(xlabel); ax.legend(fontsize=8)
    plt.tight_layout()
    out = 'assets/figures/results_greedy_vs_rl.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    evaluate()
