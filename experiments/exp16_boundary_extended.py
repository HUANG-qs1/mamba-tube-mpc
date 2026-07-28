# -*- coding: utf-8 -*-
"""exp16: 包络边界曲线的场景普遍性（AR(1) phi=0.95 + 零样本square）"""
import numpy as np
import torch
import sim_core_np as sc

PRED_LEN=10; SEQ_LEN=100; SEEDS=list(range(10)); WARMUP=200
ALPHAS=[0.7,0.85,1.0,1.15,1.3]
MU_D=-0.1; SIG_D=0.05
_state={'prev':MU_D}

def ar1_disturbance(t, amp=1.0):
    phi=0.95; eps=SIG_D*np.sqrt(1-phi**2)
    d=MU_D+phi*(_state['prev']-MU_D)+eps*np.random.randn()
    _state['prev']=d; return [amp*d,0]

class ScaledMA:
    def __init__(self, alpha, pred_len=PRED_LEN, win=10):
        self.alpha=alpha; self.pred_len=pred_len; self.win=win
    def __call__(self, x):
        xn=x.detach().cpu().numpy()[0]
        phys=xn*sc.x_std+sc.x_mean
        p=np.repeat(phys[-self.win:].mean(axis=0,keepdims=True),self.pred_len,axis=0)*self.alpha
        return torch.FloatTensor((p-sc.y_mean)/(sc.y_std+1e-8)).unsqueeze(0).to(sc.device)

def run_one(ref, alpha, seed):
    _state['prev']=MU_D
    errors,tubes,_=sc.run_sim(ref,seed,'model',model=ScaledMA(alpha),seq_len=SEQ_LEN,
        disturb='random',amp=1.0,drift=1.0,total_time=100.0)
    rmse,emax,viol=sc.metric_vs_015(errors,WARMUP)
    ea=errors[WARMUP:]; en=np.hypot(ea[:,0],ea[:,1])
    cov=1.0-np.mean(en>tubes[WARMUP:])
    return rmse, viol, cov, float(np.mean(tubes[WARMUP:]))

orig=sc.get_random_disturbance
for scene in ('ar1','square'):
    if scene=='ar1':
        sc.get_random_disturbance=ar1_disturbance; ref=sc.figure8_path(a=3.0,n=1000)
    else:
        sc.get_random_disturbance=orig; ref=sc.square_path(side=2.0,speed=0.5)
    print(f"########## {scene} ##########",flush=True)
    out={}
    for a in ALPHAS:
        rows=[run_one(ref,a,s) for s in SEEDS]
        out[a]=np.array(rows)
        m=rows and np.array(rows)
        print(f"alpha={a:4.2f}  RMSE {m[:,0].mean():.4f}+-{m[:,0].std():.4f}  "
              f"viol015 {m[:,1].mean():.1f}  cov {m[:,2].mean():.4f}  tube {m[:,3].mean():.4f}",flush=True)
    np.savez(f'exp16_{scene}.npz',**{f'alpha_{a}':v for a,v in out.items()})
sc.get_random_disturbance=orig
print("saved -> exp16_ar1.npz, exp16_square.npz",flush=True)
