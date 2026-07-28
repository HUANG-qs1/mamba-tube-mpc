# -*- coding: utf-8 -*-
"""exp15_envelope_boundary.py - 包络有效性边界扫描（路径A数值层）
sizing law 只消费预测峰值范数 => 用缩放因子 alpha 控制峰值标定偏差，
扫描 alpha in [0.6..1.5]，刻画覆盖率崩塌阈值与管宽-RMSE权衡。
预测: alpha<1 覆盖率下降; alpha>=1 覆盖100%; RMSE随alpha温和上升。
"""
import numpy as np
import torch
import sim_core_np as sc

PRED_LEN = 10
SEQ_LEN = 100
SEEDS = list(range(10))
WARMUP = 200
ALPHAS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]


class ScaledMA:
    """ma10预测，峰值范数缩放alpha倍（模拟预测器峰值标定偏差）"""
    def __init__(self, alpha, pred_len=PRED_LEN, win=10):
        self.alpha = alpha; self.pred_len = pred_len; self.win = win

    def __call__(self, x):
        xn = x.detach().cpu().numpy()[0]
        phys = xn * sc.x_std + sc.x_mean
        p = np.repeat(phys[-self.win:].mean(axis=0, keepdims=True),
                      self.pred_len, axis=0) * self.alpha
        out = (p - sc.y_mean) / (sc.y_std + 1e-8)
        return torch.FloatTensor(out).unsqueeze(0).to(sc.device)


def run_one(alpha, seed):
    ref = sc.figure8_path(a=3.0, n=1000)
    errors, tubes, _ = sc.run_sim(
        ref, seed, 'model', model=ScaledMA(alpha), seq_len=SEQ_LEN,
        disturb='random', amp=1.0, drift=1.0, total_time=100.0)
    rmse, emax, viol015 = sc.metric_vs_015(errors, WARMUP)
    ea = errors[WARMUP:]; en = np.hypot(ea[:, 0], ea[:, 1])
    cov_own = 1.0 - np.mean(en > tubes[WARMUP:])
    tube_mean = float(np.mean(tubes[WARMUP:]))
    return dict(rmse=rmse, viol015=viol015, cov_own=cov_own, tube_mean=tube_mean)


def main():
    print(f"{'alpha':>6s} {'RMSE':>16s} {'viol015':>8s} {'cov_own':>8s} {'tube':>8s}", flush=True)
    all_rows = {}
    for a in ALPHAS:
        rows = [run_one(a, s) for s in SEEDS]
        rm = np.array([r['rmse'] for r in rows])
        v15 = np.array([r['viol015'] for r in rows])
        cov = np.array([r['cov_own'] for r in rows])
        tb = np.array([r['tube_mean'] for r in rows])
        all_rows[a] = np.array([[r['rmse'], r['viol015'], r['cov_own'], r['tube_mean']] for r in rows])
        print(f"{a:6.2f} {rm.mean():.4f}+-{rm.std():.4f} {v15.mean():8.1f} "
              f"{cov.mean():8.4f} {tb.mean():8.4f}", flush=True)
    np.savez('exp15_envelope_boundary.npz', **{f'alpha_{a}': v for a, v in all_rows.items()})
    print("saved -> exp15_envelope_boundary.npz", flush=True)


if __name__ == '__main__':
    main()
