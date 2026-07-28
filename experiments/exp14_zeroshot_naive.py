# -*- coding: utf-8 -*-
"""exp14_zeroshot_naive.py - 免训练预测器的零样本泛化对照（补§5.6证据缺口）
ma10/persistence 在 square 和 lissajous 未训练轨迹上的表现 vs mamba（同种子参照）。
"""
import numpy as np
import torch
from scipy import stats

import sim_core_np as sc

PRED_LEN = 10
SEQ_LEN = 100
SEEDS = list(range(10))
WARMUP = 200


class NaivePredictor:
    def __init__(self, kind='ma10', pred_len=PRED_LEN, win=10):
        self.kind = kind; self.pred_len = pred_len; self.win = win

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


def run_one(ref, mode, seed, model=None):
    errors, tubes, _ = sc.run_sim(
        ref, seed, mode, model=model, seq_len=SEQ_LEN,
        disturb='random', amp=1.0, drift=1.0, total_time=100.0)
    rmse, emax, viol015 = sc.metric_vs_015(errors, WARMUP)
    ea = errors[WARMUP:]; en = np.hypot(ea[:, 0], ea[:, 1])
    cov_own = 1.0 - np.mean(en > tubes[WARMUP:])
    tube_mean = float(np.mean(tubes[WARMUP:]))
    return dict(rmse=rmse, viol015=viol015, cov_own=cov_own, tube_mean=tube_mean)


def main():
    paths = {
        'square':    sc.square_path(side=2.0, speed=0.5),
        'lissajous': sc.lissajous_path(A=2.0, B=2.0),
    }
    configs = [('mamba','mamba',None),
               ('naive-ma10','model',NaivePredictor('ma10', win=10)),
               ('naive-persist','model',NaivePredictor('persistence'))]
    for pname, ref in paths.items():
        print(f"########## {pname} ##########", flush=True)
        results = {}
        for name, mode, model in configs:
            rows = []
            for seed in SEEDS:
                r = run_one(ref, mode, seed, model)
                rows.append(r)
                print(f"  {name:13s} seed={seed}  RMSE={r['rmse']:.4f}  "
                      f"viol015={r['viol015']}  cov={r['cov_own']:.4f}  tube={r['tube_mean']:.4f}", flush=True)
            results[name] = rows
            rm = np.array([r['rmse'] for r in rows])
            print(f"  == {name:13s} RMSE {rm.mean():.4f}+-{rm.std():.4f}", flush=True)
        a = np.array([r['rmse'] for r in results['mamba']])
        for name in ('naive-ma10','naive-persist'):
            b = np.array([r['rmse'] for r in results[name]])
            t, p = stats.ttest_ind(a, b, equal_var=False)
            dc = (a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2)
            print(f"  Welch t  mamba vs {name}: t={t:.3f}  p={p:.3e}  Cohen's d={dc:.3f}", flush=True)
        np.savez(f'exp14_{pname}.npz',
                 **{n: np.array([[r['rmse'], r['viol015'], r['cov_own'], r['tube_mean']] for r in rows])
                    for n, rows in results.items()})
        print(flush=True)
    print("saved -> exp14_square.npz, exp14_lissajous.npz", flush=True)


if __name__ == '__main__':
    main()
