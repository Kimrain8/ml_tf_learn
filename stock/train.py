import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from config import Config
from model import StockTransformer
from dataset import StockDataset
from IPython import embed

def prepare_dummy_data(config):
    """生成模拟数据用于演示，实际使用时替换为真实数据"""
    n_samples = 2000
    features = np.random.randn(n_samples, config.history_days, config.stock_feat_dim)
    industry_id = np.random.randint(0, config.num_industries, n_samples)
    product_id = np.random.randint(0, config.num_products, n_samples)
    competitor_ids = np.random.randint(0, config.total_stocks, (n_samples, config.max_competitors))
    # 随机未来收益率
    future_return = np.random.uniform(-0.05, 0.05, n_samples)
    data = {
        'features': features,
        'industry_id': industry_id,
        'product_id': product_id,
        'competitor_ids': competitor_ids,
        'future_return': future_return
    }
    return data

def main():
    cfg = Config()
    device = cfg.device

    # 数据加载 (这里用虚拟数据演示)
    data = prepare_dummy_data(cfg)
    dataset = StockDataset(cfg, data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    # 模型、优化器、损失函数
    model = StockTransformer(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # 分类损失（加权，防止类别不平衡）
    class_weights = torch.tensor([1.0, 1.5, 1.2], device=device)  # 示例权重
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    # 回归损失
    mse_loss = nn.MSELoss()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            feat, ind, prod, comp, trend, ret = [x.to(device) for x in batch]
            optimizer.zero_grad()
            trend_logits, amount_pred = model(feat, ind, prod, comp)
            # 趋势损失
            loss_cls = ce_loss(trend_logits, trend)
            # 额度损失
            loss_reg = mse_loss(amount_pred.squeeze(), ret)
            loss = loss_cls + cfg.lambda_reg * loss_reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证（简单计算趋势准确率）
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                feat, ind, prod, comp, trend, ret = [x.to(device) for x in batch]
                trend_logits, _ = model(feat, ind, prod, comp)
                pred = torch.argmax(trend_logits, dim=1)
                correct += (pred == trend).sum().item()
                total += trend.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1:02d} | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")

if __name__ == "__main__":
    main()
    # embed()