import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaDisturbancePredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, pred_len=10):
        super().__init__()
        self.pred_len = pred_len
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.mamba1 = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.mamba2 = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim * pred_len)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.dropout(self.norm1(x + self.mamba1(x)))
        x = self.dropout(self.norm2(x + self.mamba2(x)))
        last = x[:, -1, :]
        out = self.output_proj(last)
        out = out.view(-1, self.pred_len, 2)
        return out

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    model = MambaDisturbancePredictor(input_dim=2, hidden_dim=64, output_dim=2, pred_len=10).to(device)
    x = torch.randn(4, 100, 2).to(device)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
