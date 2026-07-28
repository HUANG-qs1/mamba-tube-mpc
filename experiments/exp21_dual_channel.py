# -*- coding: utf-8 -*-
"""exp21_dual_channel.py - 双通道扰动 (d_v + d_omega) 下的预测器消融
回应评审 M3 方案B：检验"预测器无关性"是否单通道简化场景的特例。
d_v 与场景2随机剖面同分布 N(-0.1, 0.05)；d_omega ~ N(0, 0.05) rad/s，独立。
预测器不重训（Mamba/LSTM 仅在单通道语料训练）——正是评审质疑的分布外情形。
协议与正文一致：8字形、种子0-9、1000步、前200步预热弃用。
"""
import numpy as np
import torch
from scipy import stats

import sim_core_np as sc

# ---- 猴子补丁：让 RobotFree.step 的 omega 通道接受扰动 d[1] ----
def _step_dual(self, v, omega, d=None):
    x, y, theta = self.state; dt = self.dt
    d_v = d[0] if d is not None else 0.0
    d_w = d[1] if d is not None else 0.0
    v_act = v + d_v
    xn = x + v_act * np.cos(theta) * dt
    yn = y + v_act * np.sin(theta) * dt
    w_act = omega + d_w
    tn = np.arctan2(np.sin(theta + w_act * dt), np.cos(theta + w_act * dt))
    self.state = np.array([xn, yn, tn])
    return self.state
sc.RobotFree.step = _step_dual

PRED_LEN = 10
SEQ_LEN = 100
SEEDS = list(range(10))
WARMUP = 200
MU_D, SIG_D = -0.1, 0.05
SIG_W = 0.05


def dual_disturbance(t, amp=1.0):
    d_v = MU_D + SIG_D * np.random.randn()
    d_w = SIG_W * np.random.randn()
    return [amp * d_v, amp * d_w]


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


def sanity_check():
    ref = sc.figure8_path(a=3.0, n=1000)
    sc.get_random_disturbance = lambda t, amp=1.0: [0.0, 0.0]
    e0, _, _ = sc.run_sim(ref, 0, 'standard', total_time=5.0)
    sc.get_random_disturbance = lambda t, amp=1.0: [0.0, 0.5]
    e1, _, _ = sc.run_sim(ref, 0, 'standard', total_time=5.0)
    diff = float(np.abs(e1 - e0).max())
    print(f"[sanity] d_omega通道对跟踪误差的影响: max|diff| = {diff:.4f} m", flush=True)
    assert diff > 1e-3, "d_omega 未进入动力学！请检查 RobotFree.step 是否使用 d[1]"
    sc.get_random_disturbance = dual_disturbance
    print("[sanity] 通过，d_omega 已确认进入动力学。\n", flush=True)


def tost_paired(a, b, delta=0.001):
    d = a - b; n = len(d); m = float(d.mean()); se = d.std(ddof=1) / np.sqrt(n)
    p = max(1 - stats.t.cdf((m + delta) / se, n - 1),
            1 - stats.t.cdf((delta - m) / se, n - 1))
    return m, p


def main():
    sc.get_random_disturbance = dual_disturbance
    sanity_check()

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
        cov = np.array([r['cov_own'] for r in rows])
        tb = np.array([r['tube_mean'] for r in rows])
        print(f"== {name:14s} RMSE {rm.mean():.4f}+-{rm.std():.4f}  "
              f"cov_own {cov.mean():.4f}  tube {tb.mean():.4f}\n", flush=True)

    rm_mamba = np.array([r['rmse'] for r in results['mamba']])
    print("=" * 70)
    for name in ('lstm', 'naive-persist', 'naive-ma10'):
        rm_n = np.array([r['rmse'] for r in results[name]])
        tw, pw = stats.ttest_ind(rm_mamba, rm_n, equal_var=False)
        tp, pp = stats.ttest_rel(rm_mamba, rm_n)
        d_cohen = (rm_mamba.mean() - rm_n.mean()) / np.sqrt((rm_mamba.var() + rm_n.var()) / 2)
        md, ptost = tost_paired(rm_mamba, rm_n)
        print(f"mamba vs {name}:")
        print(f"  Welch   t={tw:+.3f}  p={pw:.3e}  Cohen's d={d_cohen:+.3f}")
        print(f"  paired  t={tp:+.3f}  p={pp:.3e}")
        print(f"  TOST(±1mm) 配对差={md*1000:+.2f} mm  p={ptost:.3e}", flush=True)

    np.savez('exp21_dual_channel.npz',
             **{name: np.array([[r['rmse'], r['emax'], r['viol015'], r['viol_own'],
                                 r['cov_own'], r['tube_mean']] for r in rows])
                for name, rows in results.items()})
    print("saved -> exp21_dual_channel.npz", flush=True)


if __name__ == '__main__':
    main()
