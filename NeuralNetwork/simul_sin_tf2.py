import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class XPosEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        # x: [B, L, 1]
        B, L, _ = x.shape
        pe = torch.zeros(B, L, self.d_model, device=x.device, dtype=x.dtype)
        arg = x * self.div_term[None, None, :]   # 广播 [B,L,1] x [1,1,32] = [B,L,32]
        pe[..., 0::2] = torch.sin(arg)
        pe[..., 1::2] = torch.cos(arg)
        return pe


class TransformerSinModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pe = XPosEncoding(d_model)
        # ✅ 重点：batch_first=True，消除警告
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.0,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_seq):
        # x_seq: [B, L, 1]
        emb = self.embed(x_seq)
        pos = self.pe(x_seq)
        emb = emb + pos
        feat = self.encoder(emb)
        out = self.head(feat)
        return out


# # 数据
# x = np.linspace(0, 4 * np.pi, 400)
# y_true = np.sin(x)
# x_tensor = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(-1).to(device)
# y_tensor = torch.from_numpy(y_true).float().unsqueeze(0).unsqueeze(-1).to(device)

# model = TransformerSinModel(d_model=64, nhead=4, num_layers=2).to(device)
# opt = torch.optim.Adam(model.parameters(), lr=8e-5)
# loss_fn = nn.MSELoss()

# for epoch in range(15000):
#     opt.zero_grad()
#     pred = model(x_tensor)
#     loss = loss_fn(pred, y_tensor)
#     loss.backward()
#     opt.step()
#     if epoch % 1000 == 0:
#         print(f"epoch {epoch}, loss: {loss.item():.6f}")

# with torch.no_grad():
#     y_pred = model(x_tensor).cpu().numpy().squeeze()



# 数据 0 ~ 4π
x = np.linspace(0, 4 * np.pi, 500)
y_true = np.sin(x)
x_tensor = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(-1).to(device)
y_tensor = torch.from_numpy(y_true).float().unsqueeze(0).unsqueeze(-1).to(device)

model = TransformerSinModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(8000):
    opt.zero_grad()
    pred = model(x_tensor)
    loss = loss_fn(pred, y_tensor)
    loss.backward()
    opt.step()
    if epoch % 1000 == 0:
        print(f"epoch {epoch}, loss: {loss.item():.6f}")


# embed()
torch.save(model.state_dict(), "D:/pyprj/cv2/NeuralNetwork/sin_tf2.pth")
# 推理绘图
x2 = np.linspace(0, 6 * np.pi, 700)
y2 = np.sin(x2)
x_tensor2 = torch.from_numpy(x2).float().unsqueeze(0).unsqueeze(-1).to(device)
y_tensor2 = torch.from_numpy(y2).float().unsqueeze(0).unsqueeze(-1).to(device)
with torch.no_grad():
    y_pred = model(x_tensor2).cpu().numpy().squeeze()

plt.figure(figsize=(12,5))
plt.plot(x2, y2, label="ground truth sin(x)", c="#222")
plt.plot(x2, y_pred, label="transformer pred", c="#ffb833", lw=2)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Transformer fit sin(x) from 0 to 4π (non-autoregressive)")
plt.show()

