import numpy as np
from scipy import stats

def tost(d, m=0.001):
    n = len(d); md = d.mean(); se = d.std(ddof=1)/np.sqrt(n)
    tc = stats.t.ppf(0.95, n-1)
    pU = stats.ttest_1samp(d,  m, alternative='less').pvalue
    pL = stats.ttest_1samp(d, -m, alternative='greater').pvalue
    return md*1000, (md-tc*se)*1000, (md+tc*se)*1000, max(pU, pL)

print("### exp12 phi=0.95 (10 seeds) ###")
z = np.load('exp12_ar1_correlated.npz')
print({k: z[k].shape for k in z.files})
rm = {k: z[k][:, 0] for k in z.files}
A = rm.get('mamba')
for k in ('lstm', 'naive-persist', 'naive-ma10'):
    if k in rm:
        t, p = stats.ttest_ind(A, rm[k], equal_var=False)
        md, lo, hi, pt = tost(A - rm[k])
        print(f"A vs {k}: means {A.mean():.4f} vs {rm[k].mean():.4f}  Welch p={p:.3e} | "
              f"paired {md:+.2f}mm 90%CI [{lo:+.2f},{hi:+.2f}] TOST p={pt:.4f}")

for f, phi in [('exp13_phi05.npz', 0.5), ('exp13_phi07.npz', 0.7)]:
    print(f"### {f} phi={phi} ###")
    z = np.load(f)
    print({k: z[k].shape for k in z.files})
    rm = {k: z[k][:, 0] for k in z.files}
    ks = list(rm)
    if len(ks) == 2:
        a2, b2 = rm[ks[0]], rm[ks[1]]
        t, p = stats.ttest_ind(a2, b2, equal_var=False)
        dp = (a2.mean()-b2.mean())/np.sqrt((a2.std()**2+b2.std()**2)/2)
        print(f"{ks[0]} {a2.mean():.4f} vs {ks[1]} {b2.mean():.4f}  Welch p={p:.3e}  d={abs(dp):.2f}")
