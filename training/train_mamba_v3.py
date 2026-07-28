import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from mamba_predictor import MambaDisturbancePredictor

# ========== 加载数据 ==========
print("📂 加载 training_data_v3.npz...")
data = np.load('training_data_v3.npz')
X_all = data['X_train']
Y_all = data['Y_train']

print(f"总样本: {len(X_all)}")

# 划分训练/验证集 (80/20)
n_total = len(X_all)
n_train = int(n_total * 0.8)
indices = np.random.permutation(n_total)
train_idx = indices[:n_train]
val_idx = indices[n_train:]

X_train = X_all[train_idx]
Y_train = Y_all[train_idx]
X_val = X_all[val_idx]
Y_val = Y_all[val_idx]

print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")

# ========== 标准化 ==========
x_mean = X_train.mean(axis=(0, 1), keepdims=True)
x_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
y_mean = Y_train.mean(axis=(0, 1), keepdims=True)
y_std = Y_train.std(axis=(0, 1), keepdims=True) + 1e-8

X_train_norm = (X_train - x_mean) / x_std
Y_train_norm = (Y_train - y_mean) / y_std
X_val_norm = (X_val - x_mean) / x_std
Y_val_norm = (Y_val - y_mean) / y_std

# 保存标准化参数
np.savez('norm_params_v3.npz', x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
print("✅ 标准化参数保存: norm_params_v3.npz")

# ========== Dataset ==========
class DisturbanceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(DisturbanceDataset(X_train_norm, Y_train_norm), batch_size=64, shuffle=True)
val_loader = DataLoader(DisturbanceDataset(X_val_norm, Y_val_norm), batch_size=64)

# ========== 训练 Mamba ==========
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🚀 使用设备: {device}")

model = MambaDisturbancePredictor(input_dim=2, hidden_dim=128, output_dim=2, pred_len=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()

best_val_loss = float('inf')
patience = 10
patience_counter = 0

print("=== Training Mamba v3 (with random disturbances) ===")
for epoch in range(50):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1}/50: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_mamba_v3.pt')
        patience_counter = 0
        print(f"  ✅ 保存最佳模型, Val Loss={val_loss:.6f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  ⏹️ 早停于 Epoch {epoch+1}")
            break

print(f"\n✅ Mamba v3 训练完成! Best Val Loss={best_val_loss:.6f}")
print("模型保存: best_mamba_v3.pt")

# ========== 验证：反标准化后的 RMSE ==========
model.load_state_dict(torch.load('best_mamba_v3.pt', map_location=device))
model.eval()

with torch.no_grad():
    X_val_tensor = torch.FloatTensor(X_val_norm).to(device)
    pred_norm = model(X_val_tensor).cpu().numpy()
    pred = pred_norm * y_std + y_mean
    rmse = np.sqrt(np.mean((pred - Y_val)**2))
    print(f"验证集反标准化 RMSE: {rmse:.6f}")
