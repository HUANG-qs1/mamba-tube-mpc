"""实验8：扰动失配幅度扫描（回应"κ=0 RMSE更优"质疑的核心实验）
设计：场景2轨迹（figure8）+ 复合扰动
  扫描1 幅值：amp ∈ {0.5, 1.0, 1.5, 2.0, 3.0}（drift=1）
  扫描2 漂移：drift ∈ {0.5, 2.0, 4.0}（amp=1，drift=1已在扫描1中）
方法：Mamba自适应Tube（w_base=0.02, w_min=0.08, κ=1.0）vs 固定Tube 0.05 vs 固定Tube 0.10
指标：RMSE / 覆盖率（err≤tube的步数占比）/ 平均Tube宽度 —— 核心产出是"覆盖率 vs 失配幅度"退化曲线
预期：固定0.05在失配大时覆盖率崩塌，固定0.10在失配小时过宽，自适应全档位维持高覆盖
输出：exp8_results.npz"""
import numpy as np
from sim_core_np import figure8_path, run_sim, metric_vs_tube

fig8 = figure8_path()
ALL = {}

def run_group(tag, amp, drift):
    for mname, mkw in [('adaptive', dict(mode='mamba')),
                       ('fixed_005', dict(mode='fixed', fixed_tube=0.05)),
                       ('fixed_010', dict(mode='fixed', fixed_tube=0.10))]:
        rl, rs, cl, tl = [], [], [], []
        for seed in range(10):
            e, tu, _ = run_sim(fig8, seed, disturb='composite', amp=amp, drift=drift, **mkw)
            r, mx, vi = metric_vs_tube(e, tu, 200)
            en = np.hypot(e[200:,0], e[200:,1])
            rl.append(r); rs.append(r)
            cl.append(1.0 - np.mean(en > tu[200:]))
            tl.append(np.mean(tu[200:]))
        ALL[f'{tag}_{mname}'] = {'rmse': np.mean(rl), 'rmse_std': np.std(rs),
                                 'coverage': np.mean(cl), 'tube': np.mean(tl)}
        print(f"{tag} {mname}: RMSE={np.mean(rl):.4f}±{np.std(rs):.4f} "
              f"Cov={np.mean(cl)*100:.1f}% Tube={np.mean(tl):.3f}", flush=True)

print("===== 实验8：失配幅度扫描 =====", flush=True)
for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
    run_group(f'amp{amp}', amp, 1.0)
for drift in [0.5, 2.0, 4.0]:
    run_group(f'drift{drift}', 1.0, drift)

np.savez('exp8_results.npz', results=np.array([ALL], dtype=object))
print("\n===== 实验8完成，已保存 exp8_results.npz =====", flush=True)
