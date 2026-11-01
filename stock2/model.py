import torch
import torch.nn as nn

from tokenizers import StockDataTokenizer


class StockTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.check_model_ready()
        self.config = config

        self.tokenizer = StockDataTokenizer(config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        self.cls_norm = nn.LayerNorm(config.d_model)
        self.trend_head = nn.Linear(config.d_model, config.num_classes)
        self.amount_head = nn.Linear(config.d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, stock_feat, industry_id, product_id, competitor_ids, return_attn=False):
        tokens, src_key_padding_mask = self.tokenizer(
            stock_feat, industry_id, product_id, competitor_ids
        )
        encoded = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        cls_out = self.cls_norm(encoded[:, 0, :])
        trend_logits = self.trend_head(cls_out)
        amount = self.amount_head(cls_out)

        if return_attn:
            # 用最后一层 self_attn 对 encoded 再算一次 CLS 注意力
            # 含义：「最后一层输出中，CLS 在聚合信息时最关注谁」
            # 包含前序所有层的语义，可用于分析 行业/产品/竞对/patch 的重要性
            last_self_attn = self.encoder.layers[-1].self_attn
            _, attn = last_self_attn(
                encoded, encoded, encoded,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=True,   # 8 个 head 求平均
            )
            attn = attn[:, 0, :]                 # (B, T)，CLS → 各 token
            return trend_logits, amount, attn

        return trend_logits, amount