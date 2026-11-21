# baseline.py
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

def run_baselines():
    feat_dir = "data/processed/m3_features"
    X_tab = np.load(os.path.join(feat_dir, "X_tab.npy"))
    y_gap = np.load(os.path.join(feat_dir, "y_gap.npy"))
    y_gap_excess = np.load(os.path.join(feat_dir, "y_gap_excess.npy"))
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))

    dates = sorted(meta["trade_date"].unique())
    n = len(dates)
    train_end = int(n * 0.6); val_end = int(n * 0.8)
    test_mask = meta["trade_date"].isin(dates[val_end:]).values
    train_mask = meta["trade_date"].isin(dates[:train_end]).values
    test_dates = dates[val_end:]

    print("=" * 60)
    print("训练目标：截面超额跳空收益 y_excess")
    print("=" * 60)

    X_tr = X_tab[train_mask]
    y_tr = y_gap_excess[train_mask]  # ★ 训练用超额收益
    X_te = X_tab[test_mask]
    y_te_excess = y_gap_excess[test_mask]
    
    # 训练树模型
    model = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, n_jobs=8, verbose=-1)
    model.fit(X_tr, y_tr)
    
    # 测试集预测
    pred_excess = model.predict(X_te)
    
    # 评估 1: 截面 Rank IC (每天计算预测值和真实超额收益的 Spearman 相关系数)
    test_meta = meta[test_mask].copy()
    test_meta["pred"] = pred_excess
    test_meta["true_excess"] = y_te_excess
    
    ic_list = []
    for d in test_dates:
        day_data = test_meta[test_meta["trade_date"] == d]
        if len(day_data) > 10:
            ic, _ = spearmanr(day_data["pred"], day_data["true_excess"])
            ic_list.append(ic)
    
    mean_ic = np.mean(ic_list)
    print(f"LGBM 截面 Rank IC: {mean_ic:.4f} (通常 > 0.03 就说明有选股能力)")

    # 评估 2: 如果每天只买模型预测排名前 10% 的股票，它们的绝对跳空收益是多少？
    # 对比：如果全买，平均绝对收益是多少？
    print("\n模拟选股策略（选取预测超额排名前 10% 的股票）：")
    strategy_returns = []
    baseline_returns = []
    for d in test_dates:
        day_data = test_meta[test_meta["trade_date"] == d]
        if len(day_data) == 0: continue
        
        baseline_returns.append(day_data["y_gap"].mean())
        top_10_pct = day_data.nlargest(int(len(day_data) * 0.1), "pred")
        strategy_returns.append(top_10_pct["y_gap"].mean())

    strat_mean = np.mean(strategy_returns) * 1e4
    base_mean = np.mean(baseline_returns) * 1e4
    print(f"  - 市场平均开盘跳空 (全买): {base_mean:.1f} bp")
    print(f"  - 策略选股开盘跳空 (Top10%): {strat_mean:.1f} bp")
    print(f"  - 创造 Alpha: {strat_mean - base_mean:.1f} bp")

if __name__ == "__main__":
    run_baselines()