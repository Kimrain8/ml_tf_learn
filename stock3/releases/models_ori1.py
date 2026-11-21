# models.py
import torch
import torch.nn as nn
# import torch.nn.functional as F

class LightweightFusion(nn.Module):
    # def __init__(self, seq_dim=11, tab_dim=16, cat_vocab_size=350):
    def __init__(self, seq_dim=11, tab_dim=16, cat_vocab_size=400):
        super().__init__()
        # 1. 时序分支 (TCN/1D-CNN)
        self.seq_conv = nn.Sequential(
            nn.Conv1d(in_channels=seq_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        self.seq_pool = nn.AdaptiveAvgPool1d(1) # [B, 32, L] -> [B, 32, 1]

        # 2. 类别分支 (行业 Embedding)
        self.cat_emb = nn.Embedding(cat_vocab_size, 8)

        # 3. 表格分支 (MLP)
        # 输入维度 = 16 (连续特征) + 8 (行业 embedding)
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim + 8, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # 4. 融合与输出头
        self.fusion_dim = 32 + 32  # 时序池化 + 表格输出
        self.head_mu = nn.Linear(self.fusion_dim, 1)      # 预测跳空幅度
        # self.head_sigma = nn.Linear(self.fusion_dim, 1)  # 预测 log(sigma)
        self.head_dir = nn.Linear(self.fusion_dim, 1)     # 预测方向 logit
        self.head_amp = nn.Linear(self.fusion_dim, 1)     # 预测振幅

    def forward(self, x_seq, x_tab, x_cat):
        # x_seq: [B, T, F] -> [B, F, T] for Conv1d
        x_seq = x_seq.transpose(1, 2)
        s = self.seq_conv(x_seq)          # [B, 32, T]
        s = self.seq_pool(s).squeeze(-1)  # [B, 32]

        # 类别特征并入表格
        c = self.cat_emb(x_cat.squeeze(-1)) # [B, 8]
        t = torch.cat([x_tab, c], dim=-1)   # [B, 16+8]
        t = self.tab_mlp(t)                 # [B, 32]

        # 融合
        fused = torch.cat([s, t], dim=-1)   # [B, 64]

        # 输出
        mu = self.head_mu(fused)
        # log_sigma = self.head_sigma(fused).clamp(-4, 2) # 限制范围防爆炸
        dir_logit = self.head_dir(fused)
        amp = self.head_amp(fused)
        
        # return mu, log_sigma, dir_logit, amp
        return mu, dir_logit, amp