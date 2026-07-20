import torch
import torch.nn as nn

class LSTMDisturbancePredictor(nn.Module):
    """
    LSTM baseline for comparison
    输入: (batch, seq_len, 2)
    输出: (batch, pred_len, 2)
    """
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, pred_len=10):
        super().__init__()
        self.pred_len = pred_len
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 2层 LSTM
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim * pred_len)
        
    def forward(self, x):
        x = self.input_proj(x)
        
        x1, _ = self.lstm1(x)
        x = self.dropout(self.norm1(x + x1))
        
        x2, _ = self.lstm2(x)
        x = self.dropout(self.norm2(x + x2))
        
        last = x[:, -1, :]
        out = self.output_proj(last)
        out = out.view(-1, self.pred_len, 2)
        return out

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMDisturbancePredictor(input_dim=2, hidden_dim=128, output_dim=2, pred_len=10).to(device)
    x = torch.randn(4, 100, 2).to(device)
    y = model(x)
    print("Input:", x.shape, "Output:", y.shape, "Params:", sum(p.numel() for p in model.parameters()))
