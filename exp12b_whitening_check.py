# -*- coding: utf-8 -*-
"""exp12b_whitening_check.py - Mechanism diagnostic for Section 5.9.

Measures lag-1..10 autocorrelation of (a) the injected disturbance and
(b) the closed-loop error norm, under both the iid and the AR(1) profiles.

Reported values (seeds 0-4):
  iid : disturbance lag-1 ~= 0.00, error-norm lag-1 = 0.977
  AR(1): disturbance lag-1 = 0.946, error-norm lag-1 = 0.998
The closed-loop error is strongly autocorrelated under either profile because it
is dominated by the deterministic path-induced tracking lag -- structure that a
persistence forecast already captures, which is why predictor sophistication
does not differentiate closed-loop RMSE.
"""
import numpy as np
import sim_core_np as sc

PHI, MU_D, SIG_D = 0.95, -0.1, 0.05
_state = {'prev': MU_D}

def ar1_disturbance(t, amp=1.0):
    eps = SIG_D*np.sqrt(1-PHI**2)
    d = MU_D + PHI*(_state['prev']-MU_D) + eps*np.random.randn()
    _state['prev'] = d
    return [amp*d, 0]

def autocorr(x, max_lag=10):
    x = x - x.mean()
    return [float(np.mean(x[:-k]*x[k:])/np.var(x)) for k in range(1, max_lag+1)]

def run(mode_disturb, seeds=(0,1,2,3,4)):
    orig = sc.get_random_disturbance
    if mode_disturb == 'ar1':
        sc.get_random_disturbance = ar1_disturbance
    ds, es = [], []
    for seed in seeds:
        _state['prev'] = MU_D
        ref = sc.figure8_path(a=3.0, n=1000)
        np.random.seed(seed)
        n = int(100.0/0.1)
        if mode_disturb == 'ar1':
            d_seq = [ar1_disturbance(t)[0] for t in range(n)]
        else:
            d_seq = [(-0.1 + 0.05*np.random.randn()) for t in range(n)]
        errors, _, _ = sc.run_sim(ref, seed, 'mamba', disturb='random', total_time=100.0)
        en = np.hypot(errors[200:,0], errors[200:,1])
        ds.append(autocorr(np.array(d_seq[200:])))
        es.append(autocorr(en))
    sc.get_random_disturbance = orig
    return np.mean(ds,axis=0), np.mean(es,axis=0)

for m in ('iid','ar1'):
    d_ac, e_ac = run(m)
    print(f"[{m}] disturbance autocorr lag1-5: {np.round(d_ac[:5],3)}")
    print(f"[{m}] error-norm  autocorr lag1-5: {np.round(e_ac[:5],3)}")
