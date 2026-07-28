"""实验7闭环评估：6个seq模型 × 场景2（figure8+随机扰动）× 10种子
指标：RMSE / MaxErr / Viol / 覆盖率 / 单步推理时间（关键：验证Mamba推理时间随seq_len平缓增长）
前置：需先运行 exp7_train_seq.py 生成 best_{mamba,lstm}_seq{20,100,200}.pt
输出：exp7_eval_results.npz"""
import numpy as np
import torch
from sim_core_np import figure8_path, run_sim, metric_vs_tube, time_inference, device
from mamba_predictor import MambaDisturbancePredictor
from lstm_predictor import LSTMDisturbancePredictor

fig8 = figure8_path()
models = {}
for arch, cls in [('mamba', MambaDisturbancePredictor), ('lstm', LSTMDisturbancePredictor)]:
    for seq in [20, 100, 200]:
        m = cls(2, 128, 2, 10).to(device)
        m.load_state_dict(torch.load(f'best_{arch}_seq{seq}.pt', map_location=device, weights_only=True))
        m.eval()
        models[(arch, seq)] = m
print("6个模型加载完成", flush=True)

ALL = {}
for (arch, seq), m in models.items():
    tinf = time_inference(m, seq)
    rl, ml, vl, cl = [], [], [], []
    for seed in range(10):
        e, tu, _ = run_sim(fig8, seed, 'model', model=m, seq_len=seq)
        r, mx, vi = metric_vs_tube(e, tu, 200)
        en = np.hypot(e[200:,0], e[200:,1])
        cov = 1.0 - np.mean(en > tu[200:])
        rl.append(r); ml.append(mx); vl.append(vi); cl.append(cov)
    ALL[f'{arch}_seq{seq}'] = {'rmse': np.mean(rl), 'rmse_std': np.std(rl),
                               'maxerr': np.mean(ml), 'viol': np.mean(vl),
                               'coverage': np.mean(cl), 'infer_ms': tinf}
    print(f"{arch} seq={seq}: RMSE={np.mean(rl):.4f}±{np.std(rl):.4f} MaxErr={np.mean(ml):.4f} "
          f"Viol={np.mean(vl):.1f} Cov={np.mean(cl)*100:.1f}% Infer={tinf:.2f}ms", flush=True)

np.savez('exp7_eval_results.npz', results=np.array([ALL], dtype=object))
print("\n===== 实验7闭环评估完成，已保存 exp7_eval_results.npz =====", flush=True)
