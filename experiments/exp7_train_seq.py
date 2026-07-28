"""实验7训练（v3 显存安全版）：{Mamba, LSTM} × seq_len{20, 100, 200} 共6个模型
v3 修复（针对两次 OOM）：
1. 验证/测试改为分批评估（chunk=512）——根因：全量1.1万验证样本一次上GPU，seq200时中间态约3.6GB
2. batch 降档：{20:128, 100:64, 200:32}
3. 每个 epoch 后 empty_cache 防碎片累积
4. OOM 自动减半 batch 重试（最多3次）
5. 已存在 best_xxx.pt 的模型自动跳过训练、只重估 TestMSE（断点续跑）
输出：best_{mamba,lstm}_seq{20,100,200}.pt + exp7_train_results.npz"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import os

device = 'cuda'
from mamba_predictor import MambaDisturbancePredictor
from lstm_predictor import LSTMDisturbancePredictor

d = np.load('training_data_seq200.npz')
norm = np.load('norm_params_seq200.npz')
x_mean = norm['x_mean'].squeeze(); x_std = norm['x_std'].squeeze()
y_mean = norm['y_mean'].squeeze(); y_std = norm['y_std'].squeeze()

Xtr = torch.FloatTensor((d['X_train']-x_mean)/(x_std+1e-8))
Ytr = torch.FloatTensor((d['Y_train']-y_mean)/(y_std+1e-8))
Xte = torch.FloatTensor((d['X_test']-x_mean)/(x_std+1e-8))
Yte = torch.FloatTensor((d['Y_test']-y_mean)/(y_std+1e-8))
print(f"数据加载: train={len(Xtr)} test={len(Xte)}", flush=True)

g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(Xtr), generator=g)
n_val = int(len(Xtr)*0.1)
va_idx, tr_idx = idx[:n_val], idx[n_val:]

BS_MAP = {20: 128, 100: 64, 200: 32}

def eval_mse(model, X, Y, chunk=512):
    """分批评估MSE（与nn.MSELoss()默认mean口径一致），避免全量上GPU"""
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            xb = X[i:i+chunk].to(device); yb = Y[i:i+chunk].to(device)
            n = len(xb)
            tot += F.mse_loss(model(xb), yb).item() * n
            cnt += n
    return tot / cnt

def train_with_bs(cls, seq, name, bs):
    model = cls(2, 128, 2, 10).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"\n[{name}] 参数量={nparam}（对照：Mamba=236436, LSTM=267668）batch={bs}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    tr_x, tr_y = Xtr[tr_idx][:, -seq:, :], Ytr[tr_idx]
    va_x, va_y = Xtr[va_idx][:, -seq:, :], Ytr[va_idx]
    te_x, te_y = Xte[:, -seq:, :], Yte
    best_val, best_state, bad = 1e9, None, 0
    t_start = time.time()
    for ep in range(60):
        model.train()
        perm = torch.randperm(len(tr_x))
        for i in range(0, len(tr_x), bs):
            j = perm[i:i+bs]
            xb, yb = tr_x[j].to(device), tr_y[j].to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        vl = eval_mse(model, va_x, va_y)
        torch.cuda.empty_cache()
        if vl < best_val:
            best_val, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        print(f"[{name}] epoch {ep+1} val={vl:.6f}", flush=True)
        if bad >= 10:
            print(f"[{name}] early stop at epoch {ep+1}", flush=True)
            break
    model.load_state_dict(best_state)
    tmse = eval_mse(model, te_x, te_y)
    torch.save(best_state, f'best_{name}.pt')
    mins = (time.time()-t_start)/60
    print(f"[{name}] 完成 TestMSE={tmse:.6f} 用时={mins:.1f}min 已保存 best_{name}.pt", flush=True)
    del model, opt
    torch.cuda.empty_cache()
    return tmse

def train_one(cls, seq, name):
    pt = f'best_{name}.pt'
    if os.path.exists(pt):
        model = cls(2, 128, 2, 10).to(device)
        model.load_state_dict(torch.load(pt, map_location=device, weights_only=True))
        tmse = eval_mse(model, Xte[:, -seq:, :], Yte)
        nparam = sum(p.numel() for p in model.parameters())
        print(f"\n[{name}] 检测到已训练模型，跳过训练 参数量={nparam} TestMSE={tmse:.6f}", flush=True)
        del model
        torch.cuda.empty_cache()
        return tmse
    bs = BS_MAP[seq]
    for attempt in range(3):
        try:
            return train_with_bs(cls, seq, name, bs)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(8, bs // 2)
            print(f"[{name}] OOM，batch 减半至 {bs} 重试（第{attempt+2}次）", flush=True)
    raise RuntimeError(f"{name} 多次OOM仍失败，请把日志发给Kimi")

res = {}
for arch, cls in [('mamba', MambaDisturbancePredictor), ('lstm', LSTMDisturbancePredictor)]:
    for seq in [20, 100, 200]:
        res[f'{arch}_seq{seq}'] = train_one(cls, seq, f'{arch}_seq{seq}')

mm = MambaDisturbancePredictor(2, 128, 2, 10).to(device)
mm.load_state_dict(torch.load('best_mamba_v3.pt', map_location=device, weights_only=True))
tmse_v3 = eval_mse(mm, Xte[:, -100:, :], Yte)
print(f"\n[对照] best_mamba_v3 在新测试集(seq100) TestMSE={tmse_v3:.6f}", flush=True)
res['ref_mamba_v3_on_new_test'] = tmse_v3

np.savez('exp7_train_results.npz', results=np.array([res], dtype=object))
print("\n===== 实验7训练全部完成 =====", flush=True)
for k, v in res.items():
    print(f"  {k}: TestMSE={v:.6f}", flush=True)
