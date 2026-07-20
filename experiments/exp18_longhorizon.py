# -*- coding: utf-8 -*-
"""exp18: 300s长时域包络稳定性（mamba vs ma10，回应长期漂移/递归可行性关切）"""
import numpy as np
import torch
import sim_core_np as sc

PRED_LEN=10; SEQ_LEN=100; SEEDS=list(range(10)); WARMUP=200

class NaivePredictor:
    def __init__(self, pred_len=PRED_LEN, win=10):
        self.pred_len=pred_len; self.win=win
    def __call__(self, x):
        xn=x.detach().cpu().numpy()[0]
        phys=xn*sc.x_std+sc.x_mean
        p=np.repeat(phys[-self.win:].mean(axis=0,keepdims=True),self.pred_len,axis=0)
        return torch.FloatTensor((p-sc.y_mean)/(sc.y_std+1e-8)).unsqueeze(0).to(sc.device)

def run_one(mode, seed, model=None):
    ref=sc.figure8_path(a=3.0,n=1000)
    errors,tubes,st=sc.run_sim(ref,seed,mode,model=model,seq_len=SEQ_LEN,
        disturb='random',amp=1.0,drift=1.0,total_time=300.0)
    ea=errors[WARMUP:]; en=np.hypot(ea[:,0],ea[:,1])
    cov=1.0-np.mean(en>tubes[WARMUP:])
    rmse=float(np.sqrt(np.mean(en**2)))
    # 分三段报告覆盖率，检查时间漂移
    thirds=np.array_split(np.arange(len(en)),3)
    cov_t=[1.0-np.mean(en[idx]>tubes[WARMUP:][idx]) for idx in thirds]
    return rmse, cov, cov_t, float(np.mean(tubes[WARMUP:]))

for name,mode,model in [('mamba','mamba',None),('naive-ma10','model',NaivePredictor())]:
    rows=[run_one(mode,s,model) for s in SEEDS]
    rm=np.array([r[0] for r in rows]); cv=np.array([r[1] for r in rows])
    ct=np.array([r[2] for r in rows]); tb=np.array([r[3] for r in rows])
    print(f"{name:11s} RMSE {rm.mean():.4f}+-{rm.std():.4f}  cov_total {cv.mean():.4f}  "
          f"cov_by_third {[round(float(c),4) for c in ct.mean(axis=0)]}  tube {tb.mean():.4f}",flush=True)
    np.savez(f'exp18_{name}.npz',rmse=rm,cov=cv,cov_thirds=ct,tube=tb)
print("saved -> exp18_mamba.npz, exp18_naive-ma10.npz",flush=True)
