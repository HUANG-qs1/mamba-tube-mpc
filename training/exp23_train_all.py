"""
实验2+3联合训练脚本
训练配方严格复刻 train_mamba_v3.py（bs=64, Adam lr=5e-4, MSE, 50ep, patience=10）
修正：固定seed=42，全部模型基于v4数据同一划分
实验2：pred_len in {5,10,15,20,30}
实验3：数据规模 in {10%,25%,50%,75%}（100%复用pl10模型）
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from mamba_predictor import MambaDisturbancePredictor

np.random.seed(42)
torch.manual_seed(42)
device = 'cuda'

print("加载 training_data_v4_pl30.npz...", flush=True)
data = np.load('training_data_v4_pl30.npz')
X_all, Y_all = data['X_train'], data['Y_train']
X_test, Y_test = data['X_test'], data['Y_test']
print(f"训练池: {len(X_all)}, 测试集: {len(X_test)}", flush=True)

n_total = len(X_all)
n_train = int(n_total * 0.8)
perm = np.random.RandomState(42).permutation(n_total)
tr_idx, v_idx = perm[:n_train], perm[n_train:]
X_tr_full, Y_tr_full = X_all[tr_idx], Y_all[tr_idx]
X_v, Y_v = X_all[v_idx], Y_all[v_idx]

x_mean = X_tr_full.mean(axis=(0,1), keepdims=True)
x_std = X_tr_full.std(axis=(0,1), keepdims=True) + 1e-8
y_mean = Y_tr_full.mean(axis=(0,1), keepdims=True)
y_std = Y_tr_full.std(axis=(0,1), keepdims=True) + 1e-8
np.savez('norm_params_v4.npz', x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
print("归一化参数保存: norm_params_v4.npz", flush=True)

Xtr_n = ((X_tr_full - x_mean) / x_std).astype(np.float32)
Ytr_n = ((Y_tr_full - y_mean) / y_std).astype(np.float32)
Xv_n = ((X_v - x_mean) / x_std).astype(np.float32)
Yv_n = ((Y_v - y_mean) / y_std).astype(np.float32)
Xte_n = ((X_test - x_mean) / x_std).astype(np.float32)
Yte_n = ((Y_test - y_mean) / y_std).astype(np.float32)

class DisturbanceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train_model(pred_len, sub_idx, save_path, tag):
    if sub_idx is None:
        sub_idx = np.arange(len(Xtr_n))
    X_tr = Xtr_n[sub_idx]
    Y_tr = Ytr_n[sub_idx][:, :pred_len, :]
    Yv = Yv_n[:, :pred_len, :]
    Yte = Yte_n[:, :pred_len, :]

    train_loader = DataLoader(DisturbanceDataset(X_tr, Y_tr), batch_size=64, shuffle=True)
    val_loader = DataLoader(DisturbanceDataset(Xv_n, Yv), batch_size=64)

    model = MambaDisturbancePredictor(2, 128, 2, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    best_val = float('inf'); patience = 10; cnt = 0; epochs_run = 0
    t_start = time.time()
    print(f"\n===== 开始训练 {tag} (pred_len={pred_len}, n={len(sub_idx)}) =====", flush=True)
    for epoch in range(50):
        epochs_run = epoch + 1
        model.train(); tl = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward(); optimizer.step()
            tl += loss.item()
        tl /= len(train_loader)
        model.eval(); vl = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                vl += criterion(model(bx), by).item()
        vl /= len(val_loader)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), save_path)
            cnt = 0
            mark = " *保存"
        else:
            cnt += 1
            mark = ""
        print(f"  Ep{epoch+1:02d}/50 train={tl:.6f} val={vl:.6f}{mark}", flush=True)
        if cnt >= patience:
            print(f"  早停于 Epoch {epoch+1}", flush=True)
            break
    minutes = (time.time() - t_start) / 60

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    test_loader = DataLoader(DisturbanceDataset(Xte_n, Yte), batch_size=256)
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds.append(model(bx.to(device)).cpu().numpy())
            trues.append(by.numpy())
    pred_n = np.concatenate(preds); true_n = np.concatenate(trues)
    test_mse = float(np.mean((pred_n - true_n)**2))
    pred_den = pred_n * y_std + y_mean
    true_den = Y_test[:, :pred_len, :]
    test_rmse_denorm = float(np.sqrt(np.mean((pred_den - true_den)**2)))

    x1 = torch.randn(1, 100, 2).to(device)
    with torch.no_grad():
        for _ in range(50): _ = model(x1)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(200): _ = model(x1)
        torch.cuda.synchronize()
        infer_ms = (time.time() - t0) / 200 * 1000

    rec = {'tag': tag, 'pred_len': pred_len, 'n_train': int(len(sub_idx)),
           'best_val': best_val, 'test_mse': test_mse,
           'test_rmse_denorm': test_rmse_denorm, 'infer_ms': infer_ms,
           'epochs_run': epochs_run, 'minutes': minutes}
    print(f"===== {tag} 完成: TestMSE={test_mse:.6f} RMSE={test_rmse_denorm:.4f}m "
          f"Infer={infer_ms:.3f}ms 耗时{minutes:.1f}分钟 =====", flush=True)
    return rec

exp2_results = []
for pl in [5, 10, 15, 20, 30]:
    rec = train_model(pl, None, f'best_mamba_v4_pl{pl}.pt', f'pl{pl}')
    exp2_results.append(rec)
    np.savez('exp2_pred_len_results.npz', results=np.array(exp2_results, dtype=object))

exp3_results = []
n_base = len(Xtr_n)
for scale in [0.10, 0.25, 0.50, 0.75]:
    n = int(n_base * scale)
    idx = np.random.RandomState(42).choice(n_base, n, replace=False)
    rec = train_model(10, idx, f'best_mamba_v4_scale{int(scale*100)}.pt', f'scale{int(scale*100)}')
    rec['scale'] = scale
    exp3_results.append(rec)
    np.savez('exp3_data_scale_results.npz', results=np.array(exp3_results, dtype=object))

pl10_rec = [r for r in exp2_results if r['pred_len'] == 10][0].copy()
pl10_rec['scale'] = 1.0
exp3_results.append(pl10_rec)
np.savez('exp3_data_scale_results.npz', results=np.array(exp3_results, dtype=object))

print("\n全部训练完成！", flush=True)
