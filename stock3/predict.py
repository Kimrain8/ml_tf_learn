# predict.py
import os
import numpy as np
import pandas as pd
import torch
from stock3.config import OUT_DIR
from stock3.models import LightweightFusion

def run_predict(select_ratio=0.05):
    """
    select_ratio: 选股比例。例如 0.05 挑选前 5%。
    """
    dev = torch.device("cpu")
    feat_dir = os.path.join(OUT_DIR, "m3_features")
    
    # 1. 读取最新生成的全量数据
    X_seq = np.load(os.path.join(feat_dir, "X_seq.npy"))
    X_tab = np.load(os.path.join(feat_dir, "X_tab.npy"))
    X_cat = np.load(os.path.join(feat_dir, "X_cat.npy"))
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))
    
    # 2. 自动识别 meta 中最大的日期（即数据源最新一天）
    latest_date = meta["trade_date"].max()
    
    # 提取最新一天的所有样本
    mask_latest = meta["trade_date"] == latest_date
    x_seq_t = torch.tensor(X_seq[mask_latest], dtype=torch.float32).to(dev)
    x_tab_t = torch.tensor(X_tab[mask_latest], dtype=torch.float32).to(dev)
    x_cat_t = torch.tensor(X_cat[mask_latest], dtype=torch.long).to(dev)
    
    print(f"当前数据的最新交易日: {latest_date.date()}")
    print(f"将自动基于这天的数据，生成次日选股名单。样本数: {mask_latest.sum()}")

    # 3. 加载模型
    model = LightweightFusion(seq_dim=11, tab_dim=27, cat_vocab_size=400).to(dev)
    model_path = os.path.join(OUT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}，请先运行 train.py")
        return
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.eval()
    
    # 4. 推理
    with torch.no_grad():
        mu, dir_logit = model(x_seq_t, x_tab_t, x_cat_t)
        pred_excess = mu.squeeze(-1).cpu().numpy()
        p_up = torch.sigmoid(dir_logit).squeeze(-1).cpu().numpy()

    # ★★★ 新增：限制预测值在 [-3%, +3%] 以内，防止模型“吹牛” ★★★
    pred_excess = np.clip(pred_excess, -0.03, 0.03)
    # 拼装结果
    result_df = meta[mask_latest][["symbol", "close", "pre_close"]].copy()
    result_df["pred_excess_pct"] = pred_excess * 100
    result_df["p_up"] = p_up
    
    # 排序与筛选
    result_df = result_df.sort_values("pred_excess_pct", ascending=False).reset_index(drop=True)
    result_df["rank"] = result_df.index + 1
    
    n_select = max(1, int(len(result_df) * select_ratio))
    selected_df = result_df.head(n_select)
    
    print(f"\n筛选标准：模型打分排名前 {select_ratio*100:.1f}%")
    print(f"计划筛选数量: {n_select} 只 / 全市场 {len(result_df)} 只")
    print("-" * 60)
    print(selected_df[["rank", "symbol", "close", "pred_excess_pct", "p_up"]].head(15).to_string(index=False))
    
    # 保存
    out_dir = OUT_DIR
    result_df.to_csv(os.path.join(out_dir, f"predict_all_{latest_date.date()}.csv"), index=False, encoding="utf-8-sig")
    selected_df.to_csv(os.path.join(out_dir, f"selected_top{int(select_ratio*100)}pct_{latest_date.date()}.csv"), index=False, encoding="utf-8-sig")
    
    print(f"\n[保存] 全市场预测: predict_all_{latest_date.date()}.csv")
    print(f"[保存] 选股名单: selected_top{int(select_ratio*100)}pct_{latest_date.date()}.csv")

if __name__ == "__main__":
    # 这里可以自由修改比例，比如 0.01 就是前 1%
    run_predict(select_ratio=0.05)