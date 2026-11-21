# models_v2.py
import torch
import torch.nn as nn

class LightweightFusionV2(nn.Module):
    def __init__(self, seq_dim=11, ind_seq_dim=3, tab_dim=28, cat_vocab_size=400):
        super().__init__()
        # 1. 个股时序分支 (LSTM)
        self.lstm_seq = nn.LSTM(input_size=seq_dim, hidden_size=32, num_layers=1, batch_first=True)
        
        # 2. 行业周期分支 (LSTM)
        self.lstm_ind = nn.LSTM(input_size=ind_seq_dim, hidden_size=16, num_layers=1, batch_first=True)

        # 3. 类别分支
        self.cat_emb = nn.Embedding(cat_vocab_size, 8)

        # 4. 表格分支
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim + 8, 32), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(32, 16), nn.ReLU()
        )

        # 5. 融合层
        self.fusion = nn.Sequential(
            nn.Linear(32 + 16 + 16, 32), nn.ReLU(), nn.Dropout(0.2)
        )
        self.head_mu = nn.Linear(32, 1)
        self.head_dir = nn.Linear(32, 1)

    def forward(self, x_seq, x_ind_seq, x_tab, x_cat):
        # 提取个股 LSTM 的最后一个时间步隐藏状态
        _, (h_seq, _) = self.lstm_seq(x_seq)
        s = h_seq[-1]  # [B, 32]
        
        # 提取行业 LSTM 的最后一个时间步隐藏状态
        _, (h_ind, _) = self.lstm_ind(x_ind_seq)
        i_feat = h_ind[-1] # [B, 16]
        
        c = self.cat_emb(x_cat.squeeze(-1))
        t = self.tab_mlp(torch.cat([x_tab, c], dim=-1)) # [B, 16]

        fused = torch.cat([s, i_feat, t], dim=-1)
        fused = self.fusion(fused)
        
        mu = self.head_mu(fused)
        dir_logit = self.head_dir(fused)
        return mu, dir_logit