import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from lstm_predictor import LSTMDisturbancePredictor

# 加载数据
data = np.load('training_data.npz')
X_train = data['X_train']
Y_train = data['Y_train']

norm = np.load('norm_params.npz')
x_mean = norm['x_mean'].squeeze()
x_std = norm['x_std'].squeeze()
y_mean = norm['y_mean'].squeeze()
y_std = norm['y_std'].squeeze()

X_train_norm = (X_train - x_mean) / (x_std + 1e-8)
Y_train_norm = (Y_train - y_mean) / (y_std + 1e-8)

# 划分
n = len(X_train_norm)
indices = np.random.permutation(n)
split = int(n * 0.8)
train_idx, val_idx = indices[:split], indices[split:]
X_tr, Y_tr = X_train_norm[train_idx], Y_train_norm[train_idx]
X_val, Y_val = X_train_norm[val_idx], Y_train_norm[val_idx]

class DisturbanceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(DisturbanceDataset(X_tr, Y_tr), batch_size=64, shuffle=True)
val_loader = DataLoader(DisturbanceDataset(X_val, Y_val), batch_size=64)

device = 'cuda'
model = LSTMDisturbancePredictor(input_dim=2, hidden_dim=128, output_dim=2, pred_len=10).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()

best_val_loss = float('inf')
patience = 10
patience_counter = 0

print("=== Training LSTM (hidden_dim=128) ===")
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
    
    print(f"Epoch {epoch+1} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_lstm_model.pt')
        print("  -> Saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at {epoch+1}")
            break

print(f"\nBest LSTM Val Loss: {best_val_loss:.6f}")
