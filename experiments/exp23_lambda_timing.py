# -*- coding: utf-8 -*-
"""exp23_lambda_timing.py - λ_t灵敏度扫描 + 求解计时统计 + w_k(t)抖振分析
回应次要意见：λ_t=2.0依据；IPOPT容差/迭代/失败率；w_k抖振（给作者的问题②）"""
import numpy as np
import inspect
import sim_core_np as sc

WARMUP = 200
SEEDS = list(range(10))

# ---- 参数化复制 mpc_with_tube（运行时源码替换，不动原文件） ----
src = inspect.getsource(sc.mpc_with_tube)
assert '2.0*tv**2' in src, 'lambda_t 字面量未找到，需检查 mpc_with_tube 源码'
orig_mpc = sc.mpc_with_tube

def run_mamba(seed):
    ref = sc.figure8_path(a=3.0, n=1000)
    errors, tubes, st = sc.run_sim(ref, seed, 'mamba', disturb='random',
                                   amp=1.0, drift=1.0, total_time=100.0)
    ea = errors[WARMUP:]; en = np.hypot(ea[:, 0], ea[:, 1])
    tb = tubes[WARMUP:]
    return dict(rmse=float(np.sqrt(np.mean(en**2))),
                viol=int(np.sum(en > tb)), cov=1-float(np.mean(en > tb)),
                tube=float(tb.mean()), solve_times=st, tubes=tubes, errors=en)

print('===== 第一部分: lambda_t 灵敏度扫描 =====', flush=True)
for lam in [0.5, 1.0, 2.0, 4.0, 8.0]:
    s = src.replace('2.0*tv**2', repr(float(lam)) + '*tv**2')
    s = s.replace('def mpc_with_tube', 'def mpc_lam')
    exec(s, sc.__dict__)
    sc.mpc_with_tube = sc.__dict__['mpc_lam']
    rows = [run_mamba(sd) for sd in SEEDS]
    rm = np.array([r['rmse'] for r in rows]); cv = np.array([r['cov'] for r in rows])
    tb = np.array([r['tube'] for r in rows])
    print(f'lam={lam:4.1f}  RMSE {rm.mean():.4f}+-{rm.std():.4f}  '
          f'cov {cv.mean():.4f}  tube {tb.mean():.4f}', flush=True)
sc.mpc_with_tube = orig_mpc  # 恢复

print('===== 第二部分: 计时与求解健康度（lambda=2.0, seed=0） =====', flush=True)
r = run_mamba(0)
st = r['solve_times'] * 1000.0
print(f'求解耗时(ms): mean={st.mean():.2f} p50={np.percentile(st,50):.2f} '
      f'p95={np.percentile(st,95):.2f} max={st.max():.2f} n={len(st)}', flush=True)
print(f'异常求解(>50ms)次数: {int(np.sum(st>50))}', flush=True)

print('===== 第三部分: w_k(t) 抖振分析 =====', flush=True)
w = r['tubes'][WARMUP:]
dw = np.abs(np.diff(w))
print(f'w_k: mean={w.mean():.4f} std={w.std():.4f} min={w.min():.4f} max={w.max():.4f}')
print(f'|dw|: mean={dw.mean():.5f} p95={np.percentile(dw,95):.5f} max={dw.max():.5f}')
print(f'|dw|>5mm 占比: {100*np.mean(dw>0.005):.1f}%  |dw|>10mm 占比: {100*np.mean(dw>0.01):.1f}%')
np.savez('exp23_wk_series.npz', tubes=r['tubes'], errors=r['errors'],
         solve_times=r['solve_times'])
print('saved -> exp23_wk_series.npz', flush=True)
