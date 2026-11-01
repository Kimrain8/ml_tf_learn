import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, config, data_dict):
        self.config = config
        self.features = torch.as_tensor(data_dict['features'], dtype=torch.float32)
        self.industry_id = torch.as_tensor(data_dict['industry_id'], dtype=torch.long)
        self.product_id = torch.as_tensor(data_dict['product_id'], dtype=torch.long)
        self.competitor_ids = torch.as_tensor(data_dict['competitor_ids'], dtype=torch.long)
        self.future_return = torch.as_tensor(data_dict['future_return'], dtype=torch.float32)

        ret = data_dict['future_return']
        thr = config.trend_threshold
        trend = np.ones(len(ret), dtype=np.int64)          # 默认"平"
        trend[ret > thr] = 2                               # 涨
        trend[ret < -thr] = 0                              # 跌
        self.trend_label = torch.as_tensor(trend, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.industry_id[idx],
            self.product_id[idx],
            self.competitor_ids[idx],
            self.trend_label[idx],
            self.future_return[idx],
        )