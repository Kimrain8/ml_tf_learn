import torch
import torch.nn as nn
from config import Config

class StockDataTokenizer(nn.Module):          # ← 原来是 class StockDataTokenizer:
    """
    负责将原始结构化数据转为模型可用的 token 序列和段编码。
    """
    def __init__(self, config):
        super().__init__()                    # ← 新增
        self.config = config
        # 历史 patch 投影
        self.patch_proj = nn.Linear(config.patch_len * config.stock_feat_dim, config.d_model)
        # 实体嵌入 (可学习)
        self.industry_embed = nn.Embedding(config.num_industries, config.d_model)
        self.product_embed = nn.Embedding(config.num_products, config.d_model)
        self.competitor_embed = nn.Embedding(config.total_stocks, config.d_model)  # 竞争对手用股票ID
        # 段编码 (token type): 0=CLS/SEP, 1=实体, 2=历史patch
        self.segment_embed = nn.Embedding(3, config.d_model)
        # 位置编码 (绝对位置，可学习)
        self.pos_embed = nn.Parameter(torch.randn(1, config.total_tokens, config.d_model))
        # LayerNorm
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, stock_feat, industry_id, product_id, competitor_ids):
        """
        stock_feat: (batch, history_days, stock_feat_dim)
        industry_id: (batch,)
        product_id: (batch,)
        competitor_ids: (batch, max_competitors)，值为股票ID，padding用-1
        返回: token_embeddings (batch, total_tokens, d_model)
        """
        batch_size = stock_feat.size(0)
        device = stock_feat.device

        # 1. 历史时序 -> patch tokens
        # 划分patch: (batch, days, feat) -> (batch, n_patches, patch_len, feat)
        patches = stock_feat.unfold(1, self.config.patch_len, self.config.stride)
        patches = patches.contiguous().view(batch_size, self.config.n_patches, -1)  # (B, n_patches, patch_len*feat)
        patch_tokens = self.patch_proj(patches)  # (B, n_patches, d_model)

        # 2. 实体 tokens
        ind_token = self.industry_embed(industry_id).unsqueeze(1)       # (B,1,d_model)
        prod_token = self.product_embed(product_id).unsqueeze(1)       # (B,1,d_model)

        # 竞争对手 tokens，处理 padding
        comp_mask = (competitor_ids != -1).unsqueeze(-1).float()       # (B, max_comp, 1)
        # 将 padding 的 id 替换为0（避免越界），之后用 mask 置零
        safe_comp_ids = competitor_ids.clamp(min=0)
        comp_tokens = self.competitor_embed(safe_comp_ids)            # (B, max_comp, d_model)
        comp_tokens = comp_tokens * comp_mask  # padding部分置零

        # 3. 特殊 token：[CLS] 和 [SEP]
        cls_token = torch.zeros(batch_size, 1, self.config.d_model, device=device)
        sep_token = torch.zeros(batch_size, 1, self.config.d_model, device=device)

        # 4. 拼接序列
        tokens = torch.cat([
            cls_token,
            ind_token,
            prod_token,
            comp_tokens,
            patch_tokens,
            sep_token
        ], dim=1)  # (B, total_tokens, d_model)

        # 5. 段编码
        seg_ids = torch.zeros(batch_size, tokens.size(1), dtype=torch.long, device=device)
        # CLS、SEP 保留0
        seg_ids[:, 1:1+self.config.n_entity_tokens] = 1   # 实体部分
        seg_ids[:, 1+self.config.n_entity_tokens:-1] = 2  # patch部分
        seg_emb = self.segment_embed(seg_ids)

        # 6. 位置编码
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :] + seg_emb
        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)

        return tokens