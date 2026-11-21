# backtest_v2.py
import os
import numpy as np
import pandas as pd
import torch
from config import OUT_DIR
from models_v2 import LightweightFusionV2

def run_backtest_v2(select_ratio=0.05):
    dev = torch.device("cpu")
    feat_dir = os.path.join(OUT_DIR, "m3_features_v2")
    
    X_seq = np.load(os.path.join(feat_dir, "X_seq.npy"))
    X_ind_seq = np.load(os.path.join(feat_dir, "X_ind_seq.npy"))
    X_tab = np.load(os.path.join(feat_dir, "X_tab.npy"))
    X_cat = np.load(os.path.join(feat_dir, "X_cat.npy"))
    y_gap = np.load(os.path.join(feat_dir, "y_gap.npy"))
    y_gap_excess = np.load(os.path.join(feat_dir, "y_gap_excess.npy"))
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))

    valid_mask = meta["label_valid"] == 1
    meta_valid = meta[valid_mask].copy()
    dates = sorted(meta_valid["trade_date"].unique())
    n = len(dates)
    
    test_dates = dates[int(n * 0.8):]
    test_mask = meta_valid["trade_date"].isin(test_dates).values
    
    test_meta = meta_valid[test_mask].copy()
    X_seq_test = X_seq[valid_mask][test_mask]
    X_ind_seq_test = X_ind_seq[valid_mask][test_mask]
    X_tab_test = X_tab[valid_mask][test_mask]
    X_cat_test = X_cat[valid_mask][test_mask]
    y_gap_test = y_gap[valid_mask][test_mask]
    y_excess_test = y_gap_excess[valid_mask][test_mask]

    model = LightweightFusionV2(seq_dim=11, ind_seq_dim=3, tab_dim=28, cat_vocab_size=400).to(dev)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model_v2.pth"), map_location=dev))
    model.eval()

    with torch.no_grad():
        mu, _ = model(torch.tensor(X_seq_test), torch.tensor(X_ind_seq_test), torch.tensor(X_tab_test), torch.tensor(X_cat_test))
        pred_excess = mu.squeeze(-1).numpy()

    test_meta["pred"] = pred_excess
    test_meta["true_y_gap"] = y_gap_test
    test_meta["true_excess"] = y_excess_test

    print(f"===== 回测开始 (V2: 阈值 {select_ratio*100:.1f}%) =====")
    total_top_stocks, total_top_wins = 0, 0
    total_excess_bp = 0
    daily_win = 0

    for d in test_dates:
        day_data = test_meta[test_meta["trade_date"] == d].copy()
        if len(day_data) < 10: continue

        day_data = day_data.sort_values("pred", ascending=False)
        n_sel = max(1, int(len(day_data) * select_ratio))
        top_stocks = day_data.head(n_sel)

        mkt_mean = day_data["true_y_gap"].mean()
        top_mean = top_stocks["true_y_gap"].mean()
        
        excess = (top_mean - mkt_mean) * 1e4
        total_excess_bp += excess
        
        wins = (top_stocks["true_excess"] > 0).sum()
        total_top_stocks += len(top_stocks)
        total_top_wins += wins
        
        if top_mean > mkt_mean: daily_win += 1

        print(f"日期: {d.date()} | 胜率: {wins/len(top_stocks):.1%} | 超额: {excess:+.1f}bp")

    print("\n" + "="*50)
    print(f"V2 策略日胜率: {daily_win / len(test_dates):.1%} ({daily_win}/{len(test_dates)})")
    print(f"V2 单票胜率: {(total_top_wins / total_top_stocks):.1%}")
    print(f"V2 平均每日超额: {total_excess_bp / len(test_dates):+.1f} bp")
    print("="*50)

if __name__ == "__main__":
    run_backtest_v2(select_ratio=0.05)