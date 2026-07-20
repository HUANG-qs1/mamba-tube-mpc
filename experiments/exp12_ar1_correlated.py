# -*- coding: utf-8 -*-
"""exp12_ar1_correlated.py - 时间相关扰动场景：学习型预测器 vs 朴素预测器
AR(1)有色噪声扰动，平稳分布标定为与iid场景一致(mean=-0.1, std=0.05)，
唯一变量是时间相关性 phi=0.95。零修改sim_core_np.py，猴子补丁注入。
"""
import numpy as np
import torch
from scipy import stats

import sim_core_np as sc

PRED_LEN = 10
SEQ_LEN = 100
SEEDS = list(range(10))
WARMUP = 200
PHI = 0.95          # AR(1) 相关系数，记忆长度约 1/(1-phi)*dt = 2s
MU_D = -0.1         # 平稳均值（与iid场景一致）
SIG_D = 0.05        # 平稳标准差（与iid场景一致）

# ---------- AR(1) 扰动生成器（带状态，猴子补丁替换 sc.get_random_disturbance） ----------
_ar1_state = {'prev': MU_D}

def ar1_disturbance(t, amp=1.0):
    eps_std = SIG_D * np.sqrt(1.0 - PHI**2)
    prev = _ar1_state['prev']
    d = MU_D + PHI * (prev - MU_D) + eps_std * np.random.randn()
    _ar1_state['prev'] = d
    return [amp * d, 0]

sc.get_random_disturbance = ar1_disturbance  # 注入：run_sim内部调用即变为AR(1)


class NaivePredictor:
    def __init__(self, kind='persistence', pred_len=PRED_LEN, win=10):
        assert kind in ('persistence', 'ma10')
        self.kind = kind
        self.pred_len = pred_len
        self.win = win

    def __call__(self, x):
        xn = x.detach().cpu().numpy()[0]
        phys = xn * sc.x_std + sc.x_mean
        if self.kind == 'persistence':
            p = np.repeat(phys[-1:], self.pred_len, axis=0)
        else:
            p = np.repeat(phys[-self.win:].mean(axis=0, keepdims=True),
                          self.pred_len, axis=0)
        out = (p - sc.y_mean) / (sc.y_std + 1e-8)
        return torch.FloatTensor(out).unsqueeze(0).to(sc.device)


def run_one(mode, seed, model=None):
    _ar1_state['prev'] = MU_D   # 每个run重置AR(1)状态，run_sim内部会np.random.seed(seed)保证可复现
    ref = sc.figure8_path(a=3.0, n=1000)
    errors, tubes, _ = sc.run_sim(
        ref, seed, mode, model=model, seq_len=SEQ_LEN,
        disturb='random', amp=1.0, drift=1.0, total_time=100.0)
    rmse, emax, viol015 = sc.metric_vs_015(errors, WARMUP)
    _, _, viol_own = sc.metric_vs_tube(errors, tubes, WARMUP)
    ea = errors[WARMUP:]; en = np.hypot(ea[:, 0], ea[:, 1])
    cov_own = 1.0 - np.mean(en > tubes[WARMUP:])
    tube_mean = float(np.mean(tubes[WARMUP:]))
    return dict(rmse=rmse, emax=emax, viol015=viol015,
                viol_own=viol_own, cov_own=cov_own, tube_mean=tube_mean)


def main():
    configs = [
        ('mamba',         'mamba', None),
        ('lstm',          'lstm',  None),
        ('naive-persist', 'model', NaivePredictor('persistence')),
        ('naive-ma10',    'model', NaivePredictor('ma10', win=10)),
    ]
    results = {}
    for name, mode, model in configs:
        rows = []
        for seed in SEEDS:
            r = run_one(mode, seed, model)
            rows.append(r)
            print(f"{name:14s} seed={seed}  RMSE={r['rmse']:.4f}  max={r['emax']:.3f}  "
                  f"viol015={r['viol015']}  viol_own={r['viol_own']}  "
                  f"cov_own={r['cov_own']:.4f}  tube={r['tube_mean']:.4f}", flush=True)
        results[name] = rows
        rm = np.array([r['rmse'] for r in rows])
        v15 = np.array([r['viol015'] for r in rows])
        cov = np.array([r['cov_own'] for r in rows])
        tb = np.array([r['tube_mean'] for r in rows])
        print(f"== {name:14s} RMSE {rm.mean():.4f}+-{rm.std():.4f}  "
              f"viol015 {v15.sum():.0f} (mean {v15.mean():.1f})  "
              f"cov_own {cov.mean():.4f}  tube {tb.mean():.4f}\n", flush=True)

    rm_mamba = np.array([r['rmse'] for r in results['mamba']])
    for name in ('lstm', 'naive-persist', 'naive-ma10'):
        rm_n = np.array([r['rmse'] for r in results[name]])
        t, p = stats.ttest_ind(rm_mamba, rm_n, equal_var=False)
        d_cohen = (rm_mamba.mean() - rm_n.mean()) / np.sqrt((rm_mamba.var() + rm_n.var()) / 2)
        print(f"Welch t  mamba vs {name}: t={t:.3f}  p={p:.3e}  Cohen's d={d_cohen:.3f}", flush=True)

    np.savez('exp12_ar1_correlated.npz',
             **{name: np.array([[r['rmse'], r['emax'], r['viol015'], r['viol_own'],
                                 r['cov_own'], r['tube_mean']] for r in rows])
                for name, rows in results.items()})
    print("saved -> exp12_ar1_correlated.npz", flush=True)


if __name__ == '__main__':
    main()
