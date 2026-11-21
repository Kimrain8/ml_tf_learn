# predict_v2.py
import os
import numpy as np
import pandas as pd
import torch
from config import OUT_DIR
from models_v2 import LightweightFusionV2

def run_predict_v2(select_ratio=0.05):
    dev = torch.device("cpu")
    feat_dir = os.path.join(OUT_DIR, "m3_features_v2")
    
    X_seq = np.load(os.path.join(feat_dir, "X_seq.npy"))
    X_ind_seq = np.load(os.path.join(feat_dir, "X_ind_seq.npy"))
    X_tab = np.load(os.path.join(feat_dir, "X_tab.npy"))
    X_cat = np.load(os.path.join(feat_dir, "X_cat.npy"))
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))
    
    latest_date = meta["trade_date"].max()
    print(f"当前最新交易日: {latest_date.date()}，生成预测名单...")

    mask = meta["trade_date"] == latest_date
    x_seq_t = torch.tensor(X_seq[mask], dtype=torch.float32).to(dev)
    x_ind_seq_t = torch.tensor(X_ind_seq[mask], dtype=torch.float32).to(dev)
    x_tab_t = torch.tensor(X_tab[mask], dtype=torch.float32).to(dev)
    x_cat_t = torch.tensor(X_cat[mask], dtype=torch.long).to(dev)
    
    model = LightweightFusionV2(seq_dim=11, ind_seq_dim=3, tab_dim=28, cat_vocab_size=400).to(dev)
    model_path = os.path.join(OUT_DIR, "best_model_v2.pth")
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.eval()
    
    with torch.no_grad():
        mu, dir_logit = model(x_seq_t, x_ind_seq_t, x_tab_t, x_cat_t)
        pred_excess = mu.squeeze(-1).cpu().numpy()
        p_up = torch.sigmoid(dir_logit).squeeze(-1).cpu().numpy()

    # 截断
    pred_excess = np.clip(pred_excess, -0.03, 0.03)

    result_df = meta[mask][["symbol", "close", "pre_close"]].copy()
    result_df["pred_excess_pct"] = pred_excess * 100
    result_df["p_up"] = p_up
    result_df = result_df.sort_values("pred_excess_pct", ascending=False).reset_index(drop=True)
    result_df["rank"] = result_df.index + 1
    
    n_select = max(1, int(len(result_df) * select_ratio))
    selected_df = result_df.head(n_select)
    
    print(f"\n筛选标准：前 {select_ratio*100:.1f}%")
    print(selected_df[["rank", "symbol", "close", "pred_excess_pct", "p_up"]].head(15).to_string(index=False))
    
    out_dir = OUT_DIR
    selected_df.to_csv(os.path.join(out_dir, f"selected_top{int(select_ratio*100)}pct_{latest_date.date()}_v2.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[保存] 选股名单: selected_top{int(select_ratio*100)}pct_{latest_date.date()}_v2.csv")

if __name__ == "__main__":
    run_predict_v2(select_ratio=0.05)