import numpy as np
import pandas as pd
import weakref
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utils.MyTools import MyTools
from IPython import embed

# ========== 固定随机种子，保证每次运行参数、梯度一样 ==========
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------- 1. 生成数据集 ----------------------
# x范围 [-2π, 2π]
x_np = np.linspace((1.0/2) * np.pi, 4*np.pi, 500).reshape(-1, 1)
y_np = np.sin(x_np)

# 转成torch tensor，float32
x = torch.from_numpy(x_np).float()
y = torch.from_numpy(y_np).float()


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))

class RegHook:
    """查看 model 运行时的输入输出数据, 或查看参数和梯度。
    使用方法：
    with RegHook(model, layers=[0, 3, 7], to_cpu=False) as rh:
        out = model(x)
        print(rh.intermediate.keys())
        params_info = RegHook.collect_param_and_grad(model)
    """
    def __init__(self, model, layers=None, *, to_cpu=True, name_prefix="layer"):
        self.handles = []
        self.intermediate = {}
        self.to_cpu = to_cpu
        self._closed = False
        # 1) 规范化 layers
        if layers is None:
            target = None                      # 全部
        elif isinstance(layers, int):
            target = {layers}
        elif isinstance(layers, (list, tuple, set)):
            target = set(layers)
        else:
            raise TypeError("layers must be int / list / tuple / set / None")

        # 2) 遍历模块（兼容 model.net 和 named_modules）
        if hasattr(model, "net"):
            iterable = enumerate(model.net)
        else:
            iterable = enumerate(dict(model.named_modules()).values())

        for idx, layer in iterable:
            if target is not None and idx not in target:
                continue
            cls_name = layer._get_name() if hasattr(layer, "_get_name") else type(layer).__name__
            name = f"{name_prefix}_{idx}_{cls_name}"
            self.handles.append(layer.register_forward_hook(self._make_hook(name)))
    # ---- hook 相关 ----
    def _make_hook(self, name):
        ref = weakref.ref(self)                # 弱引用，避免 self <-> handle 环
        def hook(module, inp, out):
            obj = ref()
            if obj is None or obj._closed:
                return
            obj.intermediate[name] = obj._pack(out)
        return hook

    def _pack(self, out):
        if isinstance(out, torch.Tensor):
            t = out.detach()
            return t.cpu() if self.to_cpu else t.clone()
        if isinstance(out, (list, tuple)):
            return type(out)(self._pack(o) for o in out)
        if isinstance(out, dict):
            return {k: self._pack(v) for k, v in out.items()}
        return out

    # ---- 生命周期 ----
    def close_handle(self):
        if self._closed:
            return
        self._closed = True
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close_handle()

    def __del__(self):
        try:
            self.close_handle()
        except Exception:
            pass

    # ---- 参数 / 梯度 ----
    @staticmethod
    def collect_param_and_grad(model, *, with_grad_norm=True):
        info = {}
        for name, p in model.named_parameters():
            grad = p.grad
            info[name] = {
                "param_data": p.detach().clone(),
                "grad": grad.detach().clone() if grad is not None else None,
                "grad_norm": grad.norm().item() if (with_grad_norm and grad is not None) else None,
                "shape": tuple(p.shape),
            }
        return info

# ---------------------- 2. 搭建神经网络 ----------------------
class SinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            # nn.SiLU(), # nn.ReLU(),
            nn.ReLU(),
            nn.Linear(64, 64),
            # nn.SiLU(), # nn.ReLU(),
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
    if epoch == 0:
        with RegHook(model) as rh:
            y_pred = model(x)
            print(rh.intermediate.keys())
            info_before_bwd = RegHook.collect_param_and_grad(model)
    else:
        y_pred = model(x)
    loss = criterion(y_pred, y)
    

    # 反向传播 + 更新
    optimizer.zero_grad()
    loss.backward()
    info_after_bwd = RegHook.collect_param_and_grad(model)
    # ========== 只打印【第一轮】详细信息 ==========
    # if epoch == 0:
    #     print("="*60)
    #     print(f"==== Epoch {epoch+1} 详细信息 ====")
    #     print(f"Loss value = {loss.item():.8f}")
    #     print("-"*60)
    #     print("【模型参数列表（weight / bias）】")
    #     for name, param in model.named_parameters():
    #         print(f"\n>> Param name: {name}, shape: {param.shape}")
    #         print(f"param value:\n{param.data[:3]}")  # 只打印前3行，不然太长
    #         ginfo = info_after_bwd[name]
    #         if ginfo is not None:
    #             print(f"grad norm: {ginfo['grad_norm']:.8f}")
    #             print(f"grad value:\n{ginfo['grad'][:3]}")
    #         else:
    #             print("grad: None")
    #     print("="*60)


    optimizer.step()



    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

torch.save(model.state_dict(), "D:/pyprj/cv2/NeuralNetwork/simul_sin.pth")
# new_model.load_state_dict(torch.load('final_model.pth', map_location='cpu'))

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

# embed()


# x3 = x
# sub_net = model.net[:4]
# y3 = sub_net(x3)
# y3_np = y3.detach().cpu().numpy()
# df = pd.DataFrame(data=np.column_stack((x_np[:,0], y_np,y3_np) ) ) #,columns=['x','y'])
# df.to_pickle("tst_sin3.pkl")


# x_np = np.linspace(0 * np.pi, 3 * np.pi, 100).reshape(-1, 1)
# y_np = np.sin(x_np)
# # 转成torch tensor，float32
# x4 = torch.from_numpy(x_np).float()
# y4 = torch.from_numpy(y_np).float()
# y42 = model.net(x4)
# y42 = y42.detach().cpu().numpy()

# df = pd.DataFrame(data=np.column_stack((x_np[:,0], y42) ),columns=['x','y'])
# df.to_pickle("tst_sin4.pkl")

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