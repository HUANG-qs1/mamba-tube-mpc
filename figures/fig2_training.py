# -*- coding: utf-8 -*-
"""fig2_training.py - Fig.2 v1: (a) Mamba/LSTM 训练曲线（exp7_train.log，seq200 同协议对照）
+ (b) v3 测试集单步扰动预测对比（锁定模型 best_mamba_v3 / best_lstm_model 确定性前向）。
数据纪律：不训练、不闭环仿真；曲线逐行解析自日志；预测为锁定模型在锁定测试集上的推理。
同步落盘 fig2_data.npz 备查。"""
import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import pub_style as ps

ps.apply_style()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- (a) 训练曲线：解析 exp7_train.log ----------
curves = {}
pat = re.compile(r'\[(mamba_seq200|lstm_seq200)\] epoch (\d+) val=([\d.]+)')
for line in open('exp7_train.log', encoding='utf-8', errors='ignore'):
    m = pat.search(line)
    if m:
        curves.setdefault(m.group(1), []).append((int(m.group(2)), float(m.group(3))))
for k in curves:
    curves[k].sort()
    ep0, val0 = curves[k][0]
    ep1, val1 = curves[k][-1]
    print(f"curve {k}: {len(curves[k])} epochs, first={val0:.6f}, last={val1:.6f}", flush=True)
assert 'mamba_seq200' in curves and len(curves['mamba_seq200']) >= 10, 'mamba_seq200 训练曲线行数不足'

# ---------- (b) 单步预测：锁定模型 × 锁定测试集 ----------
from mamba_predictor import MambaDisturbancePredictor
from lstm_predictor import LSTMDisturbancePredictor
norm = np.load('norm_params_v3.npz')
x_mean = norm['x_mean'].squeeze(); x_std = norm['x_std'].squeeze()
y_mean = norm['y_mean'].squeeze(); y_std = norm['y_std'].squeeze()
data = np.load('training_data_v3.npz')
X_test, Y_test = data['X_test'], data['Y_test']
print(f"test set: X{X_test.shape} Y{Y_test.shape}", flush=True)

def predict(model, X):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = (X[i:i+4096] - x_mean) / (x_std + 1e-8)
            pb = model(torch.FloatTensor(xb).to(device)).cpu().numpy()
            outs.append(pb * y_std + y_mean)
    return np.concatenate(outs)

P, MSE = {}, {}
for tag, pt, net in [('mamba', 'best_mamba_v3.pt', MambaDisturbancePredictor(2, 128, 2, 10)),
                     ('lstm', 'best_lstm_model.pt', LSTMDisturbancePredictor(2, 128, 2, 10))]:
    net.load_state_dict(torch.load(pt, map_location=device, weights_only=True))
    net.to(device).eval()
    P[tag] = predict(net, X_test)
    Yn = (Y_test - y_mean) / (y_std + 1e-8)
    Pn = (P[tag] - y_mean) / (y_std + 1e-8)
    MSE[tag] = float(np.mean((Pn - Yn) ** 2))
    print(f"full-test MSE (normalized, {tag}): {MSE[tag]:.6f}", flush=True)

# ---------- 段选：300 样本窗口中真值 d_v 方差最大者（确定性规则） ----------
SEG = 300
d0 = Y_test[:, 0, 0]
stds = np.array([d0[i:i+SEG].std() for i in range(len(d0) - SEG)])
i0 = int(np.argmax(stds))
idx = np.arange(i0, i0 + SEG)
print(f"segment: samples {i0}..{i0+SEG-1} (max-std rule, std={stds[i0]:.5f})", flush=True)

# ---------- 落盘备查 ----------
np.savez('fig2_data.npz', seg_idx=idx, seg_truth=Y_test[idx, 0, 0],
         seg_mamba=P['mamba'][idx, 0, 0], seg_lstm=P['lstm'][idx, 0, 0],
         mse_mamba=MSE['mamba'], mse_lstm=MSE['lstm'],
         **{f"curve_{k}_ep": np.array([e for e, _ in v]) for k, v in curves.items()},
         **{f"curve_{k}_val": np.array([v_ for _, v_ in v]) for k, v in curves.items()})

# ---------- 绘图 ----------
fig, (axa, axb) = plt.subplots(1, 2, figsize=ps.figsize(1.5, 0.40))
for tag, key in [('mamba_seq200', 'mamba'), ('lstm_seq200', 'lstm')]:
    if tag in curves:
        ep = [e for e, _ in curves[tag]]
        va = [v for _, v in curves[tag]]
        st = ps.style_of(key)
        st['ms'] = 2.5
        st['markevery'] = max(1, len(ep) // 12)
        axa.plot(ep, va, **st)
axa.set_xlabel('Epoch')
axa.set_ylabel('Validation MSE (normalized)')
axa.set_title('(a) training curves (seq 200)', fontsize=8, loc='left', pad=3)
axa.legend(fontsize=6.5)

def stl(key, **kw):
    s = ps.style_of(key)
    s['ms'] = 2.5
    s['markevery'] = 20
    s.update(kw)
    return s

axb.plot(idx, Y_test[idx, 0, 0], color=ps.C['black'], ls='-', lw=1.2, label='Ground truth')
axb.plot(idx, P['mamba'][idx, 0, 0], **stl('mamba'))
axb.plot(idx, P['lstm'][idx, 0, 0], **stl('lstm'))
axb.set_xlabel('Test sample index')
axb.set_ylabel(r'1-step $d_v$ (m/s)')
axb.set_title('(b) one-step prediction vs ground truth', fontsize=8, loc='left', pad=3)
axb.legend(fontsize=6.5)
fig.subplots_adjust(wspace=0.28)
os.makedirs('figures', exist_ok=True)
ps.save_fig(fig, 'figures/fig2_training')
