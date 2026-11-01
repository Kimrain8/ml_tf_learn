import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

# ---------------------- 1. 生成数据集 ----------------------
# x范围 [-2π, 2π]
x_np = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
y_np = np.sin(x_np)

# 转成torch tensor，float32
x = torch.from_numpy(x_np).float()
y = torch.from_numpy(y_np).float()

# ---------------------- 2. 搭建神经网络 ----------------------
class SinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SinNet()


class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, w0=30):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.w0 = w0
        with torch.no_grad():
            nn.init.uniform_(self.linear.weight, -np.sqrt(6/in_dim)/w0, np.sqrt(6/in_dim)/w0)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

class SirenSinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            SirenLayer(1, 64, w0=30),
            SirenLayer(64, 64, w0=1),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model2 = SirenSinNet()

# ---------------------- 3. 损失函数 & 优化器 ----------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------- 4. 训练循环 ----------------------
epochs = 5000
for epoch in range(epochs):
    # 前向传播
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # 反向传播 + 更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# ---------------------- 5. 预测 & 绘图 ----------------------
with torch.no_grad():
    y_pred = model(x)

plt.figure(figsize=(10, 5))
plt.plot(x_np, y_np, label='True sin(x)', color='blue')
plt.plot(x_np, y_pred.numpy(), label='NN predict', color='orange', linestyle='--')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("PyTorch fit sin(x)")
plt.show()

embed()


g = model.named_parameters()
w0 = next(g)[1]
w0 = w0.detach().cpu().numpy()
b0 = next(g)
b0 = b0[1].detach().cpu().numpy()
theta0 = np.column_stack((w0,b0))
x_np = x_np[x_np>0]
x = np.column_stack((x_np, np.ones_like(x_np) ))
y = x @ theta0.T
rst = []
for i in range(len(x)):
    for j in range(y.shape[1]):
        rst.append([x[i,0],y[i,j]])
import pandas as pd
df = pd.DataFrame(data=rst,columns=['x','y'])
df['y'] = df['y'].where(df['y']>0,0)
df = df.groupby(['x'])['y'].sum().reset_index()
df.to_pickle("tt.pkl")

x3 = torch.from_numpy(x[:,:1]).float()
sub_net = model.net[:5]
y3 = sub_net(x3)
y3 = y3.detach().cpu().numpy()
y3 = y3.sum(axis=1)
df = pd.DataFrame(data=np.column_stack((x[:,0], y3) ),columns=['x','y'])
df.to_pickle("tst_sin2.pkl")


x_np = np.linspace(0 * np.pi, 3 * np.pi, 100).reshape(-1, 1)
y_np = np.sin(x_np)
# 转成torch tensor，float32
x4 = torch.from_numpy(x_np).float()
y4 = torch.from_numpy(y_np).float()
y42 = model.net(x4)
y42 = y42.detach().cpu().numpy()

df = pd.DataFrame(data=np.column_stack((x_np[:,0], y42) ),columns=['x','y'])
df.to_pickle("tst_sin4.pkl")

# g = model.named_parameters()
# w0 = next(g)[1]
# w0 = w0.detach().cpu().numpy()
# b0 = next(g)
# b0 = b0[1].detach().cpu().numpy()
# theta0 = np.column_stack((w0,b0))
# x_np = x_np[x_np>0]
# x = np.column_stack((x_np, np.ones_like(x_np) ))
# y = x @ theta0.T
# rst = []
# for i in range(len(x)):
#     for j in range(y.shape[1]):
#         rst.append([x[i,0],y[i,j]])
# import pandas as pd
# df = pd.DataFrame(data=rst,columns=['x','y'])
# df.to_pickle("tt.pkl")