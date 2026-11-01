import torch
import torch.nn as nn
from config import Config
from tokenizers import StockDataTokenizer

class StockTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = StockDataTokenizer(config)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # 预测头：从 [CLS] token 输出
        self.cls_norm = nn.LayerNorm(config.d_model)
        # 趋势分类头
        self.trend_head = nn.Linear(config.d_model, config.num_classes)
        # 额度回归头
        self.amount_head = nn.Linear(config.d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, stock_feat, industry_id, product_id, competitor_ids):
        """
        输入：结构化数据
        输出：趋势logits (B,3), 预测涨跌额度 (B,1)
        """
        # 获得完整token序列
        tokens = self.tokenizer(stock_feat, industry_id, product_id, competitor_ids)
        # Transformer编码
        encoded = self.encoder(tokens)  # (B, total_tokens, d_model)
        # 提取CLS向量
        cls_out = encoded[:, 0, :]      # (B, d_model)
        cls_out = self.cls_norm(cls_out)
        # 两个预测头
        trend_logits = self.trend_head(cls_out)      # (B, 3)
        amount = self.amount_head(cls_out)           # (B, 1)
        return trend_logits, amount