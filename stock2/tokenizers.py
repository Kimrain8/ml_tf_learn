import torch
import torch.nn as nn


class StockDataTokenizer(nn.Module):
    """
    序列结构：[CLS] + [IND] + [PROD] + [COMP]*max_comp + [PATCH]*n_patches + [SEP]
    返回 (tokens, src_key_padding_mask)
    src_key_padding_mask: True 表示该位置为 padding，会被 encoder 屏蔽
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 历史 patch 投影
        self.patch_proj = nn.Linear(
            config.patch_len * config.stock_feat_dim, config.d_model
        )

        # 实体嵌入
        self.industry_embed = nn.Embedding(config.num_industries, config.d_model)
        self.product_embed = nn.Embedding(config.num_products, config.d_model)
        self.competitor_embed = nn.Embedding(config.total_stocks, config.d_model)

        # 可学习特殊 token（独立参数，不再用 zeros）
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.sep_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.pad_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # 段编码：0=CLS/SEP，1=实体，2=patch
        self.segment_embed = nn.Embedding(3, config.d_model)
        # 位置编码（可学习）
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.total_tokens, config.d_model) * 0.02
        )

        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, stock_feat, industry_id, product_id, competitor_ids):
        """
        stock_feat:     (B, history_days, stock_feat_dim)
        industry_id:    (B,)
        product_id:     (B,)
        competitor_ids: (B, max_competitors)  -1 表示 padding
        返回:
            tokens:               (B, total_tokens, d_model)
            src_key_padding_mask: (B, total_tokens) bool，True=屏蔽
        """
        B = stock_feat.size(0)
        device = stock_feat.device
        cfg = self.config

        # 1. 历史 -> patches（时间主序：patch_len 维度在后）
        patches = stock_feat.unfold(1, cfg.patch_len, cfg.stride)   # (B, n_patches, feat, patch_len)
        patches = patches.permute(0, 1, 3, 2).contiguous()          # (B, n_patches, patch_len, feat)
        patches = patches.view(B, cfg.n_patches, -1)                # (B, n_patches, patch_len*feat)
        patch_tokens = self.patch_proj(patches)                     # (B, n_patches, d_model)

        # 2. 实体 tokens
        ind_tokens = self.industry_embed(industry_id).unsqueeze(1)  # (B,1,d)
        prod_tokens = self.product_embed(product_id).unsqueeze(1)   # (B,1,d)

        comp_mask = (competitor_ids != -1)                          # (B, max_comp) True=有效
        safe_ids = competitor_ids.clamp(min=0)
        comp_tokens = self.competitor_embed(safe_ids)               # (B, max_comp, d)
        pad_tokens = self.pad_token.expand(B, competitor_ids.size(1), -1)
        comp_tokens = torch.where(comp_mask.unsqueeze(-1), comp_tokens, pad_tokens)

        # 3. 特殊 token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        sep_tokens = self.sep_token.expand(B, -1, -1)

        # 4. 拼接
        tokens = torch.cat(
            [cls_tokens, ind_tokens, prod_tokens, comp_tokens, patch_tokens, sep_tokens],
            dim=1,
        )  # (B, total_tokens, d)

        # 5. 段编码
        seg_ids = torch.zeros(B, tokens.size(1), dtype=torch.long, device=device)
        seg_ids[:, 1:1 + cfg.n_entity_tokens] = 1
        seg_ids[:, 1 + cfg.n_entity_tokens:-1] = 2
        seg_emb = self.segment_embed(seg_ids)

        # 6. 位置 + 段 + norm + dropout
        tokens = tokens + self.pos_embed + seg_emb
        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)

        # 7. key_padding_mask：只有 COMP 区间需要屏蔽
        # 序列位置：CLS(0) IND(1) PROD(2) COMP(3..3+max_comp-1) PATCH SEP
        src_key_padding_mask = torch.zeros(
            B, tokens.size(1), dtype=torch.bool, device=device
        )
        src_key_padding_mask[:, 3:3 + cfg.max_competitors] = ~comp_mask

        return tokens, src_key_padding_mask