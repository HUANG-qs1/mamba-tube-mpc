"""实验10：GP预测器对比（回应"只比LSTM不够"质疑，成本控制版）
设计（v9文档 Table 3-4）：
- GP输入特征：最近10步扰动值(20维) + 均值/方差/趋势统计量(6维) = 26维（禁止展平全序列）
- 核函数：RBF + WhiteKernel，训练集500点（GP复杂度O(n^3)的精度/成本平衡）
- d_x/d_y 各一个独立GP，递归预测10步
- 对比维度：①预测MSE（vs 同数据集训练的Mamba seq100）②单步推理时间 ③闭环RMSE（square/lissajous，vs Mamba主模型）
输出：exp10_results.npz"""
import numpy as np
import torch
import time
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
except ImportError:
    raise SystemExit("缺少scikit-learn，请先执行: pip install scikit-learn 然后重新运行")

from sim_core_np import (square_path, lissajous_path, run_sim, metric_vs_tube,
                         time_inference, device, mamba_model)
from mamba_predictor import MambaDisturbancePredictor

# ===== 数据（与实验7同一seq200数据集、同一归一化） =====
d = np.load('training_data_seq200.npz')
norm = np.load('norm_params_seq200.npz')
x_mean = norm['x_mean'].squeeze(); x_std = norm['x_std'].squeeze()
y_mean = norm['y_mean'].squeeze(); y_std = norm['y_std'].squeeze()
Xtr = (d['X_train']-x_mean)/(x_std+1e-8)
Ytr = (d['Y_train']-y_mean)/(y_std+1e-8)
Xte = (d['X_test']-x_mean)/(x_std+1e-8)
Yte = (d['Y_test']-y_mean)/(y_std+1e-8)
print(f"数据加载: train={len(Xtr)} test={len(Xte)}", flush=True)

def make_feats(buf):
    """buf:(10,2)最近10步 → 26维特征"""
    return np.concatenate([buf.flatten(), buf.mean(0), buf.std(0), (buf[-1]-buf[0])/9.0])

# ===== GP训练（500点，双GP独立） =====
rng = np.random.default_rng(0)
sel = rng.choice(len(Xtr), 500, replace=False)
Ftr = np.array([make_feats(Xtr[i][-10:]) for i in sel])
Ttr = Ytr[sel][:, 0, :]
kernel = C(1.0)*RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
gps = []
t0 = time.time()
for dim in range(2):
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gp.fit(Ftr, Ttr[:, dim])
    gps.append(gp)
gp_train_time = time.time() - t0
print(f"GP训练完成（500点×2） 用时={gp_train_time:.1f}s", flush=True)
print(f"GP核参数: {gps[0].kernel_}", flush=True)

def gp_predict_10(hist):
    """hist:(seq,2)归一化历史 → (10,2)归一化递归预测"""
    buf = hist[-10:].copy()
    out = []
    for k in range(10):
        f = make_feats(buf).reshape(1, -1)
        nx = np.array([gps[0].predict(f)[0], gps[1].predict(f)[0]])
        out.append(nx)
        buf = np.vstack([buf[1:], nx])
    return np.array(out)

class GPTubeModel:
    """与compute_tube兼容的GP包装器：输入(1,seq,2)归一化历史，输出(1,10,2)归一化预测"""
    def to(self, dev): return self
    def eval(self): return self
    def __call__(self, x):
        p = gp_predict_10(x[0].cpu().numpy())
        return torch.FloatTensor(p).unsqueeze(0).to(x.device)

gp_model = GPTubeModel()

# ===== 对照模型：同数据集训练的 Mamba seq100 =====
mm = MambaDisturbancePredictor(2, 128, 2, 10).to(device)
mm.load_state_dict(torch.load('best_mamba_seq100.pt', map_location=device, weights_only=True))
mm.eval()

# ===== 评估1：预测MSE（2000测试窗口，归一化口径） =====
print("\n===== 评估1：预测MSE对比（2000测试窗口） =====", flush=True)
n_eval = 2000
sel_te = rng.choice(len(Xte), n_eval, replace=False)
with torch.no_grad():
    mm_pred = mm(torch.FloatTensor(Xte[sel_te][:, -100:, :]).to(device)).cpu().numpy()
mm_mse = float(((mm_pred - Yte[sel_te])**2).mean())
gp_mse = 0.0
for cnt, i in enumerate(sel_te):
    gp_mse += ((gp_predict_10(Xte[i][-100:]) - Yte[i])**2).mean()
    if (cnt+1) % 500 == 0:
        print(f"  GP预测进度 {cnt+1}/{n_eval}", flush=True)
gp_mse /= n_eval
print(f"Mamba seq100 预测MSE = {mm_mse:.6f}", flush=True)
print(f"GP           预测MSE = {gp_mse:.6f}", flush=True)

# ===== 评估2：单步推理时间（完整10步预测） =====
print("\n===== 评估2：推理时间 =====", flush=True)
mm_time = time_inference(mm, 100)
gp_time = time_inference(gp_model, 100)
print(f"Mamba seq100 推理时间 = {mm_time:.2f} ms", flush=True)
print(f"GP           推理时间 = {gp_time:.2f} ms", flush=True)
print(f"GP训练时间 = {gp_train_time:.1f} s（对照：Mamba seq200 训练 22.1 min，实验7日志）", flush=True)

# ===== 评估3：闭环对比（square / lissajous × 10种子，GP-Tube vs Mamba-Tube） =====
print("\n===== 评估3：闭环对比 =====", flush=True)
ALL = {'gp_pred_mse': gp_mse, 'mamba_pred_mse': mm_mse,
       'gp_infer_ms': gp_time, 'mamba_infer_ms': mm_time,
       'gp_train_s': gp_train_time}
for pname, path in [('square', square_path()), ('lissajous', lissajous_path())]:
    for mname, mdl in [('mamba_tube', mamba_model), ('gp_tube', gp_model)]:
        rl, ml, cl = [], [], []
        for seed in range(10):
            e, tu, _ = run_sim(path, seed, 'model', model=mdl, seq_len=100)
            r, mx, vi = metric_vs_tube(e, tu, 200)
            en = np.hypot(e[200:,0], e[200:,1])
            rl.append(r); ml.append(mx); cl.append(1.0-np.mean(en > tu[200:]))
        ALL[f'{pname}_{mname}'] = {'rmse': np.mean(rl), 'rmse_std': np.std(rl),
                                   'maxerr': np.mean(ml), 'coverage': np.mean(cl)}
        print(f"{pname} {mname}: RMSE={np.mean(rl):.4f}±{np.std(rl):.4f} "
              f"MaxErr={np.mean(ml):.4f} Cov={np.mean(cl)*100:.1f}%", flush=True)

np.savez('exp10_results.npz', results=np.array([ALL], dtype=object))
print("\n===== 实验10完成，已保存 exp10_results.npz =====", flush=True)
