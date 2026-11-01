import torch


class Config:
    def __init__(self, **kwargs):
        # ---------- 数据 ----------
        self.history_days = 15
        self.patch_len = 5
        self.stride = 5
        self.future_days = 1

        # ---------- 模型 ----------
        self.d_model = 256
        self.n_head = 8
        self.n_layers = 4
        self.d_ff = 512
        self.dropout = 0.1

        # ---------- 实体（下面三个由 data_pipeline 自动回填） ----------
        self.stock_feat_dim = None
        self.num_industries = 1
        self.num_products = 1
        self.max_competitors = 5
        self.total_stocks = None

        # ---------- 预测 ----------
        self.num_classes = 3
        self.trend_threshold = 0.005

        # ---------- 训练 ----------
        self.batch_size = 64
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.epochs = 50
        self.lambda_reg = 0.5
        self.patience = 8
        self.seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._finalize()

    def _finalize(self):
        assert self.history_days >= self.patch_len
        self.n_patches = (self.history_days - self.patch_len) // self.stride + 1
        self.n_entity_tokens = 1 + 1 + self.max_competitors
        self.total_tokens = 1 + self.n_entity_tokens + self.n_patches + 1

    def check_model_ready(self):
        if self.stock_feat_dim is None:
            raise ValueError("stock_feat_dim 未设置，请先调用 build_stock_dataset()")
        if self.total_stocks is None:
            raise ValueError("total_stocks 未设置，请先调用 build_stock_dataset()")