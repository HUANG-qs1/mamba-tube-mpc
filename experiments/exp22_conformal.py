# -*- coding: utf-8 -*-
"""exp22_conformal.py - split-conformal 标定固定管基线（回应评审 M2c+M3）
校准：标准MPC在场景2族、20个校准种子(100-119)，取每轮预热后 max||e|| 为分数；
裕量：w_conf = 第 ceil((K+1)(1-alpha))/K 次序统计量（alpha=0.05，95%边际覆盖保证）；
测试：固定管 w_conf vs 自适应管(A)，场景2种子0-9同协议对比。
"""
import numpy as np
import sim_core_np as sc
from scipy import stats

WARMUP = 200
CAL_SEEDS = list(range(100, 120))
TEST_SEEDS = list(range(10))
ALPHA = 0.05

def run(mode, seed, **kw):
    ref = sc.figure8_path(a=3.0, n=1000)
    errors, tubes, _ = sc.run_sim(ref, seed, mode, disturb='random', amp=1.0,
                                  drift=1.0, total_time=100.0, **kw)
    return errors, tubes

def metrics(errors, tubes):
    ea = errors[WARMUP:]; en = np.hypot(ea[:, 0], ea[:, 1])
    rmse = float(np.sqrt(np.mean(en**2)))
    tb = tubes[WARMUP:]
    viol = int(np.sum(en > tb))
    return rmse, float(en.max()), viol, 1 - viol/len(en), float(tb.mean())

print('[1/3] 校准: 标准MPC, 20个校准种子...', flush=True)
scores = []
for s in CAL_SEEDS:
    e, _ = run('standard', s)
    ea = e[WARMUP:]
    scores.append(float(np.hypot(ea[:, 0], ea[:, 1]).max()))
    print(f'  cal seed={s} max||e||={scores[-1]:.4f}', flush=True)
scores = np.sort(np.array(scores))
K = len(scores)
rank = int(np.ceil((K + 1) * (1 - ALPHA)))
w_conf = float(scores[rank - 1])
print(f'[2/3] conformal裕量 (alpha={ALPHA}, 取第{rank}/{K}次序): w_conf = {w_conf:.4f} m', flush=True)

print('[3/3] 测试: conformal固定管 vs 自适应管(A), 种子0-9...', flush=True)
res = {}
for name, mode, kw in [('conformal-fixed', 'fixed', dict(fixed_tube=w_conf)),
                       ('adaptive-A', 'mamba', dict())]:
    rows = []
    for s in TEST_SEEDS:
        e, t = run(mode, s, **kw)
        rm, mx, vi, cv, tw = metrics(e, t)
        rows.append((rm, mx, vi, cv, tw))
        print(f'  {name:16s} seed={s} RMSE={rm:.4f} max={mx:.3f} viol={vi} '
              f'cov={cv:.4f} tube={tw:.4f}', flush=True)
    res[name] = np.array(rows)
    a = res[name]
    print(f'== {name:16s} RMSE {a[:,0].mean():.4f}+-{a[:,0].std():.4f} '
          f'viol {a[:,2].mean():.1f} cov {a[:,3].mean():.4f} tube {a[:,4].mean():.4f}', flush=True)

a, c = res['adaptive-A'][:, 0], res['conformal-fixed'][:, 0]
tw_, pw = stats.ttest_ind(a, c, equal_var=False)
tp, pp = stats.ttest_rel(a, c)
print(f'adaptive vs conformal RMSE: Welch t={tw_:+.2f} p={pw:.3e} | paired t={tp:+.2f} p={pp:.3e}')
np.savez('exp22_conformal.npz', scores=scores, w_conf=w_conf, **res)
print('saved -> exp22_conformal.npz', flush=True)
