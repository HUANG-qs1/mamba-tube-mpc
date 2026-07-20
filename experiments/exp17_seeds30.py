# -*- coding: utf-8 -*-
"""exp17: Scenario2 A/D/F/G 补跑 seed 10-29（池化成30种子，支撑TOST等效性检验）"""
import numpy as np
import torch
import sim_core_np as sc

PRED_LEN=10; SEQ_LEN=100; SEEDS=list(range(10,30)); WARMUP=200

class NaivePredictor:
    def __init__(self, kind, pred_len=PRED_LEN, win=10):
        self.kind=kind; self.pred_len=pred_len; self.win=win
    def __call__(self, x):
        xn=x.detach().cpu().numpy()[0]
        phys=xn*sc.x_std+sc.x_mean
        if self.kind=='persistence':
            p=np.repeat(phys[-1:],self.pred_len,axis=0)
        else:
            p=np.repeat(phys[-self.win:].mean(axis=0,keepdims=True),self.pred_len,axis=0)
        return torch.FloatTensor((p-sc.y_mean)/(sc.y_std+1e-8)).unsqueeze(0).to(sc.device)

def run_one(mode, seed, model=None):
    ref=sc.figure8_path(a=3.0,n=1000)
    errors,tubes,_=sc.run_sim(ref,seed,mode,model=model,seq_len=SEQ_LEN,
        disturb='random',amp=1.0,drift=1.0,total_time=100.0)
    rmse,emax,viol=sc.metric_vs_015(errors,WARMUP)
    ea=errors[WARMUP:]; en=np.hypot(ea[:,0],ea[:,1])
    cov=1.0-np.mean(en>tubes[WARMUP:])
    return rmse, viol, cov, float(np.mean(tubes[WARMUP:]))

configs=[('mamba','mamba',None),('lstm','lstm',None),
         ('naive-persist','model',NaivePredictor('persistence')),
         ('naive-ma10','model',NaivePredictor('ma10',win=10))]
res={}
for name,mode,model in configs:
    rows=[run_one(mode,s,model) for s in SEEDS]
    res[name]=np.array(rows)
    m=res[name]
    print(f"{name:14s} RMSE {m[:,0].mean():.4f}+-{m[:,0].std():.4f}  viol015 {m[:,1].mean():.1f}  "
          f"cov {m[:,2].mean():.4f}  tube {m[:,3].mean():.4f}",flush=True)
np.savez('exp17_seeds10-29.npz',**res)
print("saved -> exp17_seeds10-29.npz",flush=True)
