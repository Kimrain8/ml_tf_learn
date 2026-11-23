# predict_v2.py
import os
import glob
import numpy as np
import pandas as pd
import torch
from config import OUT_DIR
from models_v2 import LightweightFusionV2
from features_v2 import TAB_COLS  # 借用特征名列表

def get_latest_model(prefix, ext):
    """自动查找最新模型，或允许手动覆盖"""
    model_dir = os.path.join(OUT_DIR, "m3_features_v2")
    files = glob.glob(os.path.join(model_dir, f"{prefix}_*.{ext}"))
    if not files: return None
    # 按文件名（日期）倒序排，取最新的
    latest_file = sorted(files)[-1]
    print(f"自动加载最新模型: {latest_file}")
    return latest_file

def run_predict_v2(select_ratio=0.05, model_path=None):
    dev = torch.device("cpu")
    feat_dir = os.path.join(OUT_DIR, "m3_features_v2")
    
    X_seq = np.load(os.path.join(feat_dir, "X_seq.npy"))
    X_ind_seq = np.load(os.path.join(feat_dir, "X_ind_seq.npy"))
    X_tab = np.load(os.path.join(feat_dir, "X_tab.npy"))
    X_cat = np.load(os.path.join(feat_dir, "X_cat.npy"))
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))
    
    latest_date_str = str(meta["trade_date"].max()).split()[0]
    latest_date_fmt = latest_date_str.replace("-", "")
    
    # 如果没有指定模型，自动找最新的
    if model_path is None:
        model_path = get_latest_model("lstm", "pth")
        if model_path is None:
            print("未找到任何历史模型，请先运行 train_v2.py")
            return

    mask = meta["trade_date"] == meta["trade_date"].max()
    x_seq_t = torch.tensor(X_seq[mask], dtype=torch.float32).to(dev)
    x_ind_seq_t = torch.tensor(X_ind_seq[mask], dtype=torch.float32).to(dev)
    x_tab_t = torch.tensor(X_tab[mask], dtype=torch.float32).to(dev)
    x_cat_t = torch.tensor(X_cat[mask], dtype=torch.long).to(dev)
    
    model = LightweightFusionV2(seq_dim=11, ind_seq_dim=3, tab_dim=28, cat_vocab_size=400).to(dev)
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.eval()
    
    with torch.no_grad():
        mu, dir_logit = model(x_seq_t, x_ind_seq_t, x_tab_t, x_cat_t)
        pred_excess = mu.squeeze(-1).cpu().numpy()
        p_up = torch.sigmoid(dir_logit).squeeze(-1).cpu().numpy()

    pred_excess = np.clip(pred_excess, -0.03, 0.03)

    result_df = meta[mask][["symbol", "close", "pre_close"]].copy()
    result_df["pred_excess_pct"] = pred_excess * 100
    result_df["p_up"] = p_up
    result_df["today_pct"] = (result_df["close"] / result_df["pre_close"] - 1) * 100
    result_df = result_df.sort_values("pred_excess_pct", ascending=False).reset_index(drop=True)
    result_df["rank"] = result_df.index + 1
    
    n_select = max(1, int(len(result_df) * select_ratio))
    selected_df = result_df.head(n_select)
    
    print(f"\n筛选标准：前 {select_ratio*100:.1f}%")
    print(selected_df[["rank", "symbol", "close", "today_pct", "pred_excess_pct", "p_up"]].head(10).to_string(index=False))
    
    # ==========================================
    # ★ 核心功能：LSTM 归因分析 (Ablation Study)
    # ==========================================
    print("\n" + "="*50)
    print("LSTM 特征影响力分析 (前 5 名股票的特征重要性均值):")
    top_5_mask = mask.copy()
    top_5_idx = result_df.head(5).index  # 选取前5名的索引
    
    # 对表格特征做归因
    importance_tab = {}
    base_preds = pred_excess[top_5_idx]
    
    for i, col_name in enumerate(TAB_COLS):
        # 把特征列打乱（相当于消除它的影响）
        x_tab_shuffled = x_tab_t.clone()
        perm = torch.randperm(x_tab_shuffled.size(0))
        x_tab_shuffled[:, i] = x_tab_shuffled[perm, i]
        
        with torch.no_grad():
            mu_shuf, _ = model(x_seq_t, x_ind_seq_t, x_tab_shuffled, x_cat_t)
            shuf_pred = mu_shuf.squeeze(-1).cpu().numpy()[top_5_idx]
            
        # 记录预测值的变化绝对均值
        importance_tab[col_name] = np.mean(np.abs(base_preds - shuf_pred)) * 100
        
    imp_df = pd.DataFrame(list(importance_tab.items()), columns=["Feature", "Impact_Pct"]).sort_values("Impact_Pct", ascending=False)
    print(imp_df.head(5).to_string(index=False))
    imp_df.to_csv(os.path.join(OUT_DIR, f"lstm_feature_impact_{latest_date_fmt}.csv"), index=False, encoding="utf-8-sig")
    # ==========================================

    out_path = os.path.join(OUT_DIR, f"pred_lstm_{latest_date_fmt}.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] LSTM 预测名单: {out_path}")

if __name__ == "__main__":
    # 可以手动传入 model_path 如: run_predict_v2(model_path="xxx")
    run_predict_v2(select_ratio=0.05)