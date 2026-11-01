import torch

from config import Config
from model import StockTransformer
from business_classifier import classify_main_business


def load_model(ckpt_path, device=None):
    """加载 checkpoint，返回 (model, cfg, meta)"""
    # ckpt = torch.load(ckpt_path, map_location=device or 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device or 'cpu', weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt['cfg'].items()})
    cfg.device = device or torch.device('cpu')
    cfg._finalize()
    model = StockTransformer(cfg).to(cfg.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, cfg, ckpt['meta']


@torch.no_grad()
def inference(model, cfg, stock_feat, industry_id, product_id, competitor_ids):
    """
    stock_feat: (B, history_days, stock_feat_dim)  已标准化
    返回: trend_pred (B,), trend_proba (B, C), expected_return (B,)
    """
    device = cfg.device
    stock_feat = stock_feat.to(device)
    industry_id = industry_id.to(device)
    product_id = product_id.to(device)
    competitor_ids = competitor_ids.to(device)
    logits, pred_ret = model(stock_feat, industry_id, product_id, competitor_ids)
    return (
        logits.argmax(dim=1),
        torch.softmax(logits, dim=1),
        pred_ret.squeeze(-1),
    )


def industry_name_to_id(name, meta):
    """输入申万三级行业名称，返回 industry_id；未见过返回 0 并告警"""
    m = meta.get('industry_name_to_id', {})
    if name not in m:
        fallback = meta.get('industry_id_to_name', {}).get(0, '未知')
        print(f"[warn] 行业 '{name}' 不在训练集映射中，使用 id=0（{fallback}）")
        return 0
    return m[name]


def product_id_from_business(main_business, business_scope=None):
    """从主营业务文本预测 product_id（与训练时一致）"""
    return classify_main_business(main_business, business_scope)


if __name__ == '__main__':
    model, cfg, meta = load_model('checkpoints/best.pt')

    sw3_name = '白酒'
    main_business = '公司主要从事高端白酒的生产与销售，拥有知名白酒品牌。'
    biz_scope = '白酒的生产、销售；预包装食品销售。'

    ind_id = industry_name_to_id(sw3_name, meta)
    prod_id = product_id_from_business(main_business, biz_scope)
    print(
        f"industry_id={ind_id} ({sw3_name})  "
        f"product_id={prod_id} ({meta['product_class_names'][prod_id]})"
    )

    B = 1
    feat = torch.randn(B, cfg.history_days, cfg.stock_feat_dim)  # 真实数据要用 meta['mean']/['std'] 标准化
    ind = torch.tensor([ind_id], dtype=torch.long)
    prod = torch.tensor([prod_id], dtype=torch.long)
    comp = torch.full((B, cfg.max_competitors), -1, dtype=torch.long)

    trend, proba, ret = inference(model, cfg, feat, ind, prod, comp)
    print(
        f"trend={trend.item()}  proba={proba.cpu().numpy().round(3)}  "
        f"expected_return={ret.item():.4f}"
    )