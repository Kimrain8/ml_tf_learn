import torch
from torch.utils.data import Dataset
import numpy as np
from config import Config

class StockDataset(Dataset):
    def __init__(self, config, data_dict):
        """
        data_dict 需包含：
        - 'features': (N, history_days, feat_dim) 历史日线特征
        - 'industry_id': (N,) 行业ID
        - 'product_id': (N,) 产品ID
        - 'competitor_ids': (N, max_competitors) 竞争对手股票ID，padding=-1
        - 'future_return': (N,) 未来N日对数收益率
        """
        self.config = config
        self.features = torch.FloatTensor(data_dict['features'])
        self.industry_id = torch.LongTensor(data_dict['industry_id'])
        self.product_id = torch.LongTensor(data_dict['product_id'])
        self.competitor_ids = torch.LongTensor(data_dict['competitor_ids'])
        self.future_return = torch.FloatTensor(data_dict['future_return'])

        # 生成趋势标签：跌(0) 平(1) 涨(2)
        self.trend_label = torch.zeros(len(self.future_return), dtype=torch.long)
        ret = self.future_return.numpy()
        self.trend_label[ret > config.trend_threshold] = 2
        self.trend_label[ret < -config.trend_threshold] = 0
        self.trend_label[(ret >= -config.trend_threshold) & (ret <= config.trend_threshold)] = 1

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.industry_id[idx],
            self.product_id[idx],
            self.competitor_ids[idx],
            self.trend_label[idx],
            self.future_return[idx]
        )