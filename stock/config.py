import torch

class Config:
    # 数据
    stock_feat_dim = 8          # 日线特征数：开、高、低、收、成交量、换手率等
    history_days = 60           # 回溯天数
    patch_len = 5               # 每个patch包含的天数
    stride = 5                  # patch步长（不重叠）
    n_patches = history_days // stride   # 12个patch
    d_model = 256               # Transformer隐藏维度
    n_head = 8
    n_layers = 6
    d_ff = 512
    dropout = 0.1

    # 实体嵌入
    num_industries = 50         # 行业总数
    num_products = 200          # 产品类别总数
    max_competitors = 5         # 最大竞争对手数量
    total_stocks = 3000         # 股票总数，用于竞争对手ID嵌入
    entity_embed_dim = 64       # 原始实体嵌入维度，之后投影到d_model

    # 特殊token
    cls_token_id = 0
    sep_token_id = 1
    pad_token_id = 2

    # 序列构成： [CLS] + 行业 + 产品 + 竞争对手*max_competitors + 历史patches + [SEP]
    # 实体token数量：1 (行业) + 1 (产品) + max_competitors
    n_entity_tokens = 1 + 1 + max_competitors
    total_tokens = 1 + n_entity_tokens + n_patches + 1  # CLS + 实体 + patches + SEP

    # 预测目标
    future_days = 5
    trend_threshold = 0.005     # 涨跌阈值
    num_classes = 3             # 跌、平、涨

    # 训练
    batch_size = 64
    lr = 1e-4
    epochs = 50
    lambda_reg = 0.5            # 回归损失权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')