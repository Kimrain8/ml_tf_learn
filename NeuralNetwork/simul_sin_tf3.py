import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
from IPython import embed
from utils.MyTf import RegHook
"""三个实验: 1, 使用对角线mask 拟合sin(x); 2, 外推预测能力； 3, 使用时序进行预测"""


# 保证结果可复现
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 之前的 XPosEncoding (用于实验1和2)
class XPosEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        # x: [B, L, 1]
        arg = x * self.div_term.view(1, 1, -1)
        pe = torch.zeros(*x.shape[:2], self.d_model, device=x.device, dtype=x.dtype)
        pe[..., 0::2] = torch.sin(arg)
        pe[..., 1::2] = torch.cos(arg)
        return pe

# 2. 画注意力图的工具
def plot_heads(attn, batch_idx=0, title=""):
    """绘制第1批次的 的注意力权重
    attn: 需要显示的注意力权重[batch,num_headers,L,L]
    batch_idx: 默认为第1批次
    """
    attn = attn.detach().cpu()
    H = attn.size(1)
    fig, axes = plt.subplots(1, H, figsize=(4 * H, 4))
    if H == 1: axes = [axes]
    for h in range(H):
        im = axes[h].imshow(attn[batch_idx, h], aspect="auto", cmap="viridis")
        axes[h].set_title(f"head {h}")
        axes[h].set_xlabel("key pos"); axes[h].set_ylabel("query pos")
    fig.suptitle(title); fig.colorbar(im, ax=axes, fraction=0.02); plt.show()

# 3. 通用训练函数
def train_model(model, x_train, y_train, epochs=500, lr=1e-3, mask_fn=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if mask_fn:
            pred = model(x_train, mask=mask_fn(x_train.size(1), x_train.device))
        else:
            pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.6f}")
    return model

# 3. 定义对角线 Mask
def get_diag_mask(L, dev):
    mask = torch.full((L, L), float('-inf'), device=dev)
    mask.fill_diagonal_(0.0)
    return mask


print("=== 实验 1：拟合 sin(x) (对角线 Mask) ===")
# 1. 生成训练数据 (x 在 [-10, 10] 之间)
L = 200
x_train = torch.linspace(-10, 10, L).view(1, L, 1).to(device)  # [B=1, L=200, 1]
y_train = torch.sin(x_train)
# 2. 模型定义
class SinModel_Diag(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pe = XPosEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        emb = self.input_proj(x) + self.pe(x)
        x = self.encoder(emb, mask=mask)
        return self.output_proj(x)

# 生成模型实例
model1 = SinModel_Diag().to(device)
if os.path.exists("D:/pyprj/cv2/NeuralNetwork/sin_tf3.pth"):
    state_dict = torch.load('D:/pyprj/cv2/NeuralNetwork/sin_tf3.pth', map_location='cpu')
    model1.load_state_dict(state_dict)
else:
    train_model(model1, x_train, y_train, epochs=500, lr=1e-3, mask_fn=get_diag_mask)
    torch.save(model1.state_dict(), "D:/pyprj/cv2/NeuralNetwork/sin_tf3.pth")

# 4. 预测与绘图
# model1.eval()
# with torch.no_grad():
#     pred1 = model1(x_train, mask=get_diag_mask(L, device))

model1.eval()  # 切换到推理模式（防止 dropout 等干扰）
with torch.no_grad():  # 禁用梯度计算
    with RegHook(model=model1, inout='in') as rh: #layers='encoder.layers.0.self_attn', 
        pred1 = model1(x_train, mask=get_diag_mask(L, device))

embed()

attn = RegHook.get_attn_sys(model=model1,rh=rh,layers='encoder.layers.0.self_attn',attn_mask=get_diag_mask(L, device))
plot_heads(attn=attn['encoder.layers.0.self_attn'], batch_idx=0, title="Layer 0 Self-Attention")
# embed()

# L = 400
# x_train = torch.linspace(-10, 15, L).view(1, L, 1).to(device)  # [B=1, L=200, 1]
# y_train = torch.sin(x_train)
# pred1 = model1(x_train, mask=get_diag_mask(L, device))

plt.figure(figsize=(10, 4))
plt.plot(x_train[0,:,0].cpu(), y_train[0,:,0].cpu(), label='True sin(x)')
plt.plot(x_train[0,:,0].cpu(), pred1[0,:,0].detach().cpu(), '--', label='Pred')
plt.title("Exp 1: Fit within [-10, 10]"); plt.legend(); plt.show()


"""
print("\n=== 实验 2：外推能力测试 ===")

# 1. 生成更宽范围的数据 (训练只在 [-5, 5]，测试在 [-10, 10])
L = 200
x_train_ext = torch.linspace(-5, 5, L).view(1, L, 1).to(device)
x_test_ext  = torch.linspace(-10, 10, L).view(1, L, 1).to(device)

class SinModel_Fourier(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        # ★ 关键：输入维度变成 3 (x, sin(x), cos(x))
        self.input_proj = nn.Linear(3, d_model) 
        self.pe = XPosEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1) # 1层就够
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        # 构造傅里叶特征
        feats = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1) # [B, L, 3]
        emb = self.input_proj(feats) + self.pe(x)
        x_out = self.encoder(emb, mask=mask)
        return self.output_proj(x_out)

# 2. 训练
y_train_ext = torch.sin(x_train_ext)
model2 = SinModel_Fourier().to(device)
# train_model(model2, x_train_ext, y_train_ext, epochs=500, lr=1e-3, mask_fn=get_diag_mask)


if os.path.exists("D:/pyprj/cv2/NeuralNetwork/sin_tf4.pth"):
    state_dict = torch.load('D:/pyprj/cv2/NeuralNetwork/sin_tf4.pth', map_location='cpu')
    model2.load_state_dict(state_dict)
else:
    train_model(model2, x_train_ext, y_train_ext, epochs=500, lr=1e-3, mask_fn=get_diag_mask)
    torch.save(model2.state_dict(), "D:/pyprj/cv2/NeuralNetwork/sin_tf4.pth")

# 4. 预测与绘图
# model1.eval()
# with torch.no_grad():
#     pred1 = model1(x_train, mask=get_diag_mask(L, device))

model2.eval()  # 切换到推理模式（防止 dropout 等干扰）
with torch.no_grad():  # 禁用梯度计算
    with RegHook(model=model2, layers='encoder.layers.0.self_attn', inout='in') as rh:
        pred2 = model2(x_test_ext, mask=get_diag_mask(L, device))

attn = RegHook.get_attn_sys(model=model2,rh=rh,layers='encoder.layers.0.self_attn')
plot_heads(attn=attn['encoder.layers.0.self_attn'], batch_idx=0, title="Layer 0 Self-Attention")
# embed()


# 3. 测试外推
# model2.eval()
# with torch.no_grad():
#     pred2 = model2(x_test_ext, mask=get_diag_mask(L, device))

plt.figure(figsize=(10, 4))
plt.plot(x_test_ext[0,:,0].cpu(), torch.sin(x_test_ext)[0,:,0].cpu(), label='True sin(x)')
plt.plot(x_test_ext[0,:,0].cpu(), pred2[0,:,0].cpu(), '--', label='Pred (Trained on [-5,5])')
plt.axvspan(-5, 5, color='gray', alpha=0.2, label='Train Zone')
plt.title("Exp 2: Extrapolate outside training zone"); plt.legend(); plt.show()
"""




""" 
print("\n=== 实验 3：时序预测 (Causal Mask) ===")

# 1. 生成时间序列数据 (输入序列模型)
# 用前 50 个点，预测第 51 个点
seq_len = 50
t = torch.linspace(0, 20 * np.pi, 500).to(device)
y_seq = torch.sin(t)

# 构造滑动窗口数据
X_seq, Y_seq = [], []
for i in range(len(y_seq) - seq_len):
    X_seq.append(y_seq[i:i+seq_len])
    Y_seq.append(y_seq[i+seq_len])


X_seq = torch.stack(X_seq).unsqueeze(-1) # [N, 50, 1]
Y_seq = torch.stack(Y_seq).unsqueeze(-1) # [N, 1]

# 取前 400 个训练，后 50 个测试
x_tr, y_tr = X_seq[:400].to(device), Y_seq[:400].to(device)
x_te, y_te = X_seq[400:].to(device), Y_seq[400:].to(device)


def sinusoidal_pos_emb(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)

class SinModel_Seq(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.register_buffer("pos_emb", sinusoidal_pos_emb(seq_len, d_model))   # ✅ 已修
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2,
                                           dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        emb = self.input_proj(x) + self.pos_emb
        x_out = self.encoder(emb, mask=mask)
        return self.output_proj(x_out[:, -1:, :]).squeeze(-1)

model3 = SinModel_Seq().to(device)
optimizer = optim.Adam(model3.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

CKPT = "D:/pyprj/cv2/NeuralNetwork/sin_tf7.pth"     # ✅ 换名，强制重训
if os.path.exists(CKPT):
    model3.load_state_dict(torch.load(CKPT, map_location=device))
    print("Loaded existing checkpoint:", CKPT)
else:
    train_loader = DataLoader(TensorDataset(x_tr, y_tr),
                              batch_size=32, shuffle=True,
                              pin_memory=(device.type == "cuda"))
    print("Start training on", device)
    for epoch in range(1500):
        model3.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model3(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        if (epoch + 1) % 100 == 0:
            avg = total / len(x_tr)
            print(f"Epoch {epoch+1} | Loss: {avg:.6f}")
            if avg < 0.01:
                print("Converged, stop early.")
                break
    torch.save(model3.state_dict(), CKPT)
    

# model3.eval()  # 切换到推理模式（防止 dropout 等干扰）
# with torch.no_grad():  # 禁用梯度计算
#     with RegHook(model=model3, layers='encoder.layers.0.self_attn', inout='in') as rh:
#         pred_te = model3(x_te, mask=get_causal_mask(seq_len, device))
# 推理：也不加 mask
model3.eval()
with torch.no_grad():
    with RegHook(model=model3, inout='in') as rh: #layers=['encoder.layers.0.self_attn','output_proj'], 
        pred_te = model3(x_te)

embed()


attn = RegHook.get_attn_sys(model=model3,rh=rh,layers='encoder.layers.0.self_attn')
plot_heads(attn=attn['encoder.layers.0.self_attn'], batch_idx=0, title="Layer 0 Self-Attention")


# 4. 测试
# model3.eval()
# with torch.no_grad():
#     pred_te = model3(x_te, mask=get_causal_mask(seq_len, device))

plt.figure(figsize=(10, 4))
plt.plot(y_te.cpu().numpy(), label='True Future')
plt.plot(pred_te.cpu().numpy(), '--', label='Predicted Future')
plt.title("Exp 3: Sequence Prediction (Predict next step)"); plt.legend(); plt.show()
 """

