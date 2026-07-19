# -*- coding: utf-8 -*-
"""exp13_ar1_phi_scan.py - AR(1) phi sensitivity scan (supplement to exp12, Section 5.9).

phi = 0.5 / 0.7, endpoints mamba vs naive-ma10, 10 seeds, same protocol as exp12.
Purpose: show that phi = 0.95 in Section 5.9 is not a favourable parameter pick.

Reported results:
  phi = 0.5: mamba 0.0675+-0.0005 vs ma10 0.0664+-0.0005 (p = 9.4e-05, d = 2.36)
  phi = 0.7: mamba 0.0676+-0.0007 vs ma10 0.0665+-0.0006 (p = 1.8e-03, d = 1.73)
Coverage stays at 100% and mean tube widths within 0.089-0.091 m at every phi;
the ~1.6% RMSE edge of the moving average at moderate phi is reported as measured.
"""
import numpy as np
import torch
from scipy import stats

import sim_core_np as sc

PRED_LEN = 10
SEQ_LEN = 100
SEEDS = list(range(10))
WARMUP = 200
MU_D = -0.1
SIG_D = 0.05

_state = {'prev': MU_D, 'phi': 0.95}

def ar1_disturbance(t, amp=1.0):
    phi = _state['phi']
    eps_std = SIG_D * np.sqrt(1.0 - phi**2)
    d = MU_D + phi * (_state['prev'] - MU_D) + eps_std * np.random.randn()
    _state['prev'] = d
    return [amp * d, 0]

sc.get_random_disturbance = ar1_disturbance


class NaivePredictor:
    def __init__(self, kind='ma10', pred_len=PRED_LEN, win=10):
        self.kind = kind; self.pred_len = pred_len; self.win = win

    def __call__(self, x):
        xn = x.detach().cpu().numpy()[0]
        phys = xn * sc.x_std + sc.x_mean
        p = np.repeat(phys[-self.win:].mean(axis=0, keepdims=True),
                      self.pred_len, axis=0)
        out = (p - sc.y_mean) / (sc.y_std + 1e-8)
        return torch.FloatTensor(out).unsqueeze(0).to(sc.device)


def run_one(mode, seed, model=None):
    _state['prev'] = MU_D
    ref = sc.figure8_path(a=3.0, n=1000)
    errors, tubes, _ = sc.run_sim(
        ref, seed, mode, model=model, seq_len=SEQ_LEN,
        disturb='random', amp=1.0, drift=1.0, total_time=100.0)
    rmse, emax, viol015 = sc.metric_vs_015(errors, WARMUP)
    ea = errors[WARMUP:]; en = np.hypot(ea[:, 0], ea[:, 1])
    cov_own = 1.0 - np.mean(en > tubes[WARMUP:])
    tube_mean = float(np.mean(tubes[WARMUP:]))
    return dict(rmse=rmse, viol015=viol015, cov_own=cov_own, tube_mean=tube_mean)


def main():
    ma10 = NaivePredictor('ma10', win=10)
    for phi in (0.5, 0.7):
        _state['phi'] = phi
        print(f"########## phi = {phi} ##########", flush=True)
        results = {}
        for name, mode, model in [('mamba','mamba',None), ('naive-ma10','model',ma10)]:
            rows = []
            for seed in SEEDS:
                r = run_one(mode, seed, model)
                rows.append(r)
                print(f"  {name:11s} seed={seed}  RMSE={r['rmse']:.4f}  "
                      f"viol015={r['viol015']}  cov={r['cov_own']:.4f}  tube={r['tube_mean']:.4f}", flush=True)
            results[name] = rows
            rm = np.array([r['rmse'] for r in rows])
            print(f"  == {name:11s} RMSE {rm.mean():.4f}+-{rm.std():.4f}", flush=True)
        a = np.array([r['rmse'] for r in results['mamba']])
        b = np.array([r['rmse'] for r in results['naive-ma10']])
        t, p = stats.ttest_ind(a, b, equal_var=False)
        dc = (a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2)
        print(f"  Welch t  mamba vs naive-ma10: t={t:.3f}  p={p:.3e}  Cohen's d={dc:.3f}\n", flush=True)
        np.savez(f'exp13_phi{str(phi).replace(".","")}.npz',
                 **{n: np.array([[r['rmse'], r['viol015'], r['cov_own'], r['tube_mean']] for r in rows])
                    for n, rows in results.items()})
    print("saved -> exp13_phi05.npz, exp13_phi07.npz", flush=True)


if __name__ == '__main__':
    main()
