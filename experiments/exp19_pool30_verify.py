import numpy as np
from scipy import stats
a = np.load('exp11_naive_baseline.npz')   # seeds 0-9
b = np.load('exp17_seeds10-29.npz')       # seeds 10-29
def pool(name):
    return np.concatenate([a[name][:, 0], b[name][:, 0]])
A = pool('mamba'); F = pool('naive-persist'); G = pool('naive-ma10')
for nm, x in [('A mamba', A), ('F persist', F), ('G ma10', G)]:
    print(f"{nm:10s} n={len(x)}  RMSE {x.mean():.4f}+-{x.std():.4f}")
for nm, x in [('F', F), ('G', G)]:
    t, p = stats.ttest_ind(A, x, equal_var=False)
    d = A - x
    md = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    tc = stats.t.ppf(0.95, len(d) - 1)
    lo, hi = md - tc * se, md + tc * se
    print(f"A vs {nm}: Welch t={t:.3f} p={p:.3e} | paired diff {md*1000:+.2f}mm "
          f"90%CI [{lo*1000:+.2f},{hi*1000:+.2f}]mm "
          f"TOST±1mm: {'EQUIVALENT' if lo > -0.001 and hi < 0.001 else 'NOT-equiv'}")
