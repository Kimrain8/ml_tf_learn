import numpy as np
import pandas as pd
from IPython import embed
import matplotlib
matplotlib.use('QtAgg')   # 或 'TkAgg'/'Qt5Agg' 需要安装PyQt5或PyQt6 / 'MacOSX'
import matplotlib.pyplot as plt
from utils.MyTools import MyTools as mt
from utils.MyTf import RegHook

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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

class TransformerSinModel2(nn.Module):
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

class TransformerSinModel1(nn.Module):
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
            batch_first=True, #False,
            # activation=F.silu,
            activation='gelu',
            dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [B, 1]  (batch, feature)
        
        B = x.shape[0]
        emb = self.input_proj(x)          # [B, d_model]
        # x = x.unsqueeze(0)              # [seq_len=1, B, d_model]
        pos = self.pe(x)                  # 加位置编码
        emb = emb + pos
        # print(emb.shape)
        x = self.encoder(emb)             # transformer encoder [1, B, d_model]
        # print(x.shape)
        out = self.output_proj(x)
        return out



def split_heads(mha, x):
    """
    mha: nn.MultiheadAttention (batch_first=True)
    x  : [B, L, E]
    返回每头的 Q, K, V, 注意力权重
    """
    B, L, E = x.shape
    H  = mha.num_heads
    Dh = E // H
    assert E == mha.embed_dim

    # 1) 取出合并的 QKV 权重和 bias
    W = mha.in_proj_weight          # [3E, E]
    b = mha.in_proj_bias            # [3E]
    Wq, Wk, Wv = W.chunk(3, dim=0)  # 各 [E, E]
    bq, bk, bv = b.chunk(3, dim=0)  # 各 [E]

    # 2) 线性投影
    q = F.linear(x, Wq, bq)         # [B, L, E]
    k = F.linear(x, Wk, bk)
    v = F.linear(x, Wv, bv)

    # 3) 拆头
    def reshape(t):
        return t.view(B, L, H, Dh).transpose(1, 2)   # [B, H, L, Dh]

    q_h, k_h, v_h = reshape(q), reshape(k), reshape(v)

    # 4) 每头注意力
    scale  = Dh ** 0.5
    scores = (q_h @ k_h.transpose(-2, -1)) / scale   # [B, H, L, L]
    attn   = scores.softmax(dim=-1)                  # [B, H, L, L]

    return q_h, k_h, v_h, attn

def tst_last_linear():
    """测试TF模型的最后一层输入 实际上可以组装很多不同的结果"""
    model1 = TransformerSinModel1()
    model2 = TransformerSinModel2()

    state_dict = torch.load('D:/pyprj/cv2/NeuralNetwork/sin_tf1.pth', map_location='cpu')
    model1.load_state_dict(state_dict)

    state_dict = torch.load('D:/pyprj/cv2/NeuralNetwork/sin_tf2.pth', map_location='cpu')
    model2.load_state_dict(state_dict)

    x_np = np.linspace(3 * np.pi, 5 * np.pi, 500)
    y_np = np.sin(x_np)

    x_tensor = torch.from_numpy(x_np).float().unsqueeze(0).unsqueeze(-1)
    y_tensor = torch.from_numpy(y_np).float().unsqueeze(0).unsqueeze(-1)
    #下面获取倒数第2层的输出，然后检验是不是可以模拟出cos的函数
    with RegHook(model1, layers=['encoder.layers.1.norm2','encoder.layers.1.dropout2', 'output_proj'], to_cpu=False) as rh:
        out = model1(x_tensor)
        print(rh.intermediate.keys())
        params_info = RegHook.collect_param_and_grad(model1)

    arr = rh.intermediate['encoder.layers.1.norm2_out'].detach().numpy()
    # df = np.column_stack((x_np,y_np,arr[0,:,:]) )
    # df = pd.DataFrame(data=df,columns=[f"col{i}" for i in range(df.shape[1])])
    # df.to_pickle("tst_sin5.pkl")

    y = np.cos(1.5* x_np)
    w,rsd,rank,sigma = np.linalg.lstsq(arr[0,:,:],y)
    y_pred = arr[0,:,:]@w
    plt.scatter(x_np,y_pred)
    plt.show()

def tst_attn():
    
    # layer_names = [f"encoder.layers.{i}.self_attn"
    #             for i in range(model.encoder.num_layers)]
    x_np = np.linspace(3 * np.pi, 5 * np.pi, 500)
    y_np = np.sin(x_np)
    x_tensor = torch.from_numpy(x_np).float().unsqueeze(0).unsqueeze(-1)
    y_tensor = torch.from_numpy(y_np).float().unsqueeze(0).unsqueeze(-1)
    model1 = TransformerSinModel1()
    state_dict = torch.load('D:/pyprj/cv2/NeuralNetwork/sin_tf1.pth', map_location='cpu')
    model1.load_state_dict(state_dict)
    layer_names = ['encoder.layers.0.self_attn','encoder.layers.1.self_attn']


    with RegHook(model1, layers=layer_names,
                inout="in", to_cpu=False) as rh:
        _ = model1(x_tensor)

    # 每层的 self_attn 输入（Q=K=V 的那个 x）
    for i in range(model1.encoder.num_layers):
        x_in = rh.intermediate[f"encoder.layers.{i}.self_attn_in"][0]  # [B,L,E]
        mha  = model1.encoder.layers[i].self_attn
        _, _, _, attn = split_heads(mha, x_in)
        print(f"layer {i} attn: {attn.shape}")   # [B, H, L, L]


if __name__ == "__main__":
    # tst_last_linear()
    tst_attn()
    #获取encodelayer的注意力：
    
    
    # embed()