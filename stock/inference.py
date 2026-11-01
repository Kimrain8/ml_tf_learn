import torch
from config import Config
from model import StockTransformer

def inference(model, stock_feat, industry_id, product_id, competitor_ids):
    """
    返回：趋势类别(0/1/2), 预测收益率, 注意力权重（可选）
    """
    model.eval()
    with torch.no_grad():
        trend_logits, amount = model(stock_feat, industry_id, product_id, competitor_ids)
        trend_pred = torch.argmax(trend_logits, dim=1)
        trend_proba = torch.softmax(trend_logits, dim=1)
        # 如果想提取注意力权重，需要在encoder输出attn_weights，此处略
    return trend_pred, amount.squeeze(), trend_proba

# 使用示例
if __name__ == "__main__":
    cfg = Config()
    model = StockTransformer(cfg)
    model.load_state_dict(torch.load("model.pth"))
    # 单样本推理：扩展batch维度
    single_feat = torch.randn(1, cfg.history_days, cfg.stock_feat_dim)
    ind = torch.tensor([10])
    prod = torch.tensor([5])
    comp = torch.randint(0, cfg.total_stocks, (1, cfg.max_competitors))
    trend, amount, prob = inference(model, single_feat, ind, prod, comp)
    print(f"Predicted trend: {trend.item()}, Expected return: {amount.item():.4f}")