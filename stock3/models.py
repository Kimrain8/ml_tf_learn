# models.py
import torch
import torch.nn as nn

class LightweightFusion(nn.Module):
    def __init__(self, seq_dim=11, tab_dim=27, cat_vocab_size=400):
        super().__init__()
        self.seq_conv = nn.Sequential(
            nn.Conv1d(seq_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cat_emb = nn.Embedding(cat_vocab_size, 8)
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim + 8, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.fusion_dim = 16 + 16
        self.head_mu = nn.Linear(self.fusion_dim, 1)
        self.head_dir = nn.Linear(self.fusion_dim, 1)

    def forward(self, x_seq, x_tab, x_cat):
        s = self.seq_conv(x_seq.transpose(1, 2)).squeeze(-1)  # [B, 16]
        t = self.tab_mlp(torch.cat([x_tab, self.cat_emb(x_cat.squeeze(-1))], dim=-1))
        fused = torch.cat([s, t], dim=-1)
        return self.head_mu(fused), self.head_dir(fused)