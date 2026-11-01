import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

# ---------------------- 1. 数据 ----------------------
x_np = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
y_np = np.sin(x_np)
x = torch.from_numpy(x_np).float()
y = torch.from_numpy(y_np).float()

# ---------------------- 2. 位置编码（必须！Transformer没有位置感知） ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return x

class XPosEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 提前预计算div_term，注册buffer，不要每次forward都算
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        # x: [B, L, 1]  真实x坐标
        B, L, _ = x.shape
        pe = torch.zeros(B, L, self.d_model, device=x.device, dtype=x.dtype)
        # [B, L, 1] * [32] -> [B, L, 32]
        arg = x * self.div_term.unsqueeze(0).unsqueeze(0)
        pe[..., 0::2] = torch.sin(arg)
        pe[..., 1::2] = torch.cos(arg)
        return pe

# ---------------------- 3. Transformer 回归模型 ----------------------
class TransformerSinModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.d_model = d_model
        # 输入映射：标量x -> d_model维度
        self.input_proj = nn.Linear(1, d_model)
        # self.pe = PositionalEncoding(d_model)
        self.pe = XPosEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,
            dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出映射 d_model -> 1
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [B, 1]  (batch, feature)
        B = x.shape[0]
        x = self.input_proj(x)          # [B, d_model]
        x = x.unsqueeze(0)              # [seq_len=1, B, d_model]
        x = self.pe(x)                  # 加位置编码
        x = self.encoder(x)             # transformer encoder [1, B, d_model]
        out = self.output_proj(x[0])    # 取唯一token输出 [B,1]
        return out

model = TransformerSinModel(d_model=64, nhead=4, num_layers=2)

# ---------------------- 4. 训练配置 ----------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 8000

for epoch in range(epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")


# ---------------------- 5. 自增加 ----------------------
import pandas as pd
x_np = np.linspace(0 * np.pi, 4 * np.pi, 1000).reshape(-1, 1)
y_np = np.sin(x_np)
x = torch.from_numpy(x_np).float()
y = torch.from_numpy(y_np).float()
with torch.no_grad():
    y_pred = model(x)
y_pred = y_pred.detach().cpu().numpy()
df = pd.DataFrame(data=np.column_stack((x_np[:,0], y_pred) ),columns=['x','y'])
df.to_pickle("tst_sin4.pkl")

# ---------------------- 5. 绘图 ----------------------
# with torch.no_grad():
#     y_pred = model(x)

plt.figure(figsize=(10,5))
plt.plot(x_np, y_np, label="True sin(x)", c="blue")
plt.plot(x_np, y_pred.numpy(), label="Transformer predict", c="orange", linestyle="--")
plt.legend()
plt.title("Transformer fit sin(x)")
plt.show()

# embed()