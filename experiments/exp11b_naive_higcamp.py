# -*- coding: utf-8 -*-
"""exp11b_naive_higcamp.py - 朴素基线 x 高幅值扰动（自适应区检验）"""
import numpy as np
import torch
from scipy import stats

import sim_core_np as sc

PRED_LEN = 10
SEQ_LEN = 100
SEEDS = list(range(10))
WARMUP = 200
AMPS = [2.0, 3.0]


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


def run_one(mode, seed, amp, model=None):
    ref = sc.figure8_path(a=3.0, n=1000)
    errors, tubes, _ = sc.run_sim(
        ref, seed, mode, model=model, seq_len=SEQ_LEN,
        disturb='random', amp=amp, drift=1.0, total_time=100.0)
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
        ('naive-persist', 'model', NaivePredictor('persistence')),
        ('naive-ma10',    'model', NaivePredictor('ma10', win=10)),
    ]
    for amp in AMPS:
        print(f"########## amp x{amp} ##########", flush=True)
        results = {}
        for name, mode, model in configs:
            rows = []
            for seed in SEEDS:
                r = run_one(mode, seed, amp, model)
                rows.append(r)
                print(f"amp={amp} {name:14s} seed={seed}  RMSE={r['rmse']:.4f}  "
                      f"max={r['emax']:.3f}  viol015={r['viol015']}  "
                      f"viol_own={r['viol_own']}  cov_own={r['cov_own']:.4f}  "
                      f"tube={r['tube_mean']:.4f}", flush=True)
            results[name] = rows
            rm = np.array([r['rmse'] for r in rows])
            v15 = np.array([r['viol015'] for r in rows])
            vo = np.array([r['viol_own'] for r in rows])
            cov = np.array([r['cov_own'] for r in rows])
            tb = np.array([r['tube_mean'] for r in rows])
            print(f"== amp={amp} {name:14s} RMSE {rm.mean():.4f}+-{rm.std():.4f}  "
                  f"viol015 {v15.sum():.0f}  viol_own {vo.sum():.0f}  "
                  f"cov_own {cov.mean():.4f}  tube {tb.mean():.4f}\n", flush=True)

        rm_mamba = np.array([r['rmse'] for r in results['mamba']])
        for name in ('naive-persist', 'naive-ma10'):
            rm_n = np.array([r['rmse'] for r in results[name]])
            t, p = stats.ttest_ind(rm_mamba, rm_n, equal_var=False)
            print(f"Welch t  amp={amp} mamba vs {name}: t={t:.3f}  p={p:.3e}",
                  flush=True)

        np.savez(f'exp11b_amp{amp}.npz',
                 **{name: np.array([[r['rmse'], r['emax'], r['viol015'],
                                     r['viol_own'], r['cov_own'], r['tube_mean']]
                                    for r in rows])
                    for name, rows in results.items()})
        print(f"saved -> exp11b_amp{amp}.npz\n", flush=True)


if __name__ == '__main__':
    main()
