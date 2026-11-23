# baseline_v2.py
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb
from config import OUT_DIR

def run_baselines():
    print("=" * 60)
    print("基线模型评估 (基于 V2 最新特征，纯树模型不打时序)")
    print("=" * 60)
    
    feat_dir = os.path.join(OUT_DIR, "m3_features_v2")
    X_tab = np.load(os.path.join(feat_dir, "X_tab.npy"))
    X_cat = np.load(os.path.join(feat_dir, "X_cat.npy"))
    y_gap = np.load(os.path.join(feat_dir, "y_gap.npy"))
    y_gap_excess = np.load(os.path.join(feat_dir, "y_gap_excess.npy"))
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))

    # 拼接表格和行业 ID 特征
    X_all = np.hstack([X_tab, X_cat])

    # ★ 必须严格过滤有效标签（防止最后一天/无效样本混入训练）
    valid_mask = meta["label_valid"] == 1
    meta_valid = meta[valid_mask].copy()
    
    dates = sorted(meta_valid["trade_date"].unique())
    n = len(dates)
    train_end = int(n * 0.6); val_end = int(n * 0.8)
    test_dates = dates[val_end:]

    train_mask = meta_valid["trade_date"].isin(dates[:train_end]).values
    val_mask = meta_valid["trade_date"].isin(dates[train_end:val_end]).values
    test_mask = meta_valid["trade_date"].isin(dates[val_end:]).values

    # 提取数据
    X_tr = X_all[valid_mask][train_mask]
    y_tr = y_gap_excess[valid_mask][train_mask]
    
    X_va = X_all[valid_mask][val_mask]
    y_va = y_gap_excess[valid_mask][val_mask]
    
    X_te = X_all[valid_mask][test_mask]
    y_te_excess = y_gap_excess[valid_mask][test_mask]
    
    # 训练树模型（开启早停，防止过拟合）
    model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, 
        n_jobs=8, verbose=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(20, verbose=False)])
    
    # 测试集预测
    pred_excess = model.predict(X_te)
    
    # 评估 1: 截面 Rank IC
    test_meta = meta_valid[test_mask].copy()
    test_meta["pred"] = pred_excess
    test_meta["true_excess"] = y_te_excess
    test_meta["true_gap"] = y_gap[valid_mask][test_mask]
    
    ic_list = []
    for d in test_dates:
        day_data = test_meta[test_meta["trade_date"] == d]
        if len(day_data) > 10:
            ic, _ = spearmanr(day_data["pred"], day_data["true_excess"])
            if not np.isnan(ic): ic_list.append(ic)
    
    mean_ic = np.mean(ic_list)
    print(f"基线 LGBM 截面 Rank IC: {mean_ic:.4f} (通常 > 0.03 就说明有选股能力)")

    # 评估 2: 模拟选股策略（Top 10%）
    print("\n模拟选股策略（选取预测超额排名前 10% 的股票）：")
    strategy_returns = []
    baseline_returns = []
    for d in test_dates:
        day_data = test_meta[test_meta["trade_date"] == d]
        if len(day_data) == 0: continue
        
        baseline_returns.append(day_data["true_gap"].mean())
        top_10_pct = day_data.nlargest(int(len(day_data) * 0.1), "pred")
        strategy_returns.append(top_10_pct["true_gap"].mean())

    strat_mean = np.mean(strategy_returns) * 1e4
    base_mean = np.mean(baseline_returns) * 1e4
    print(f"  - 市场平均开盘跳空 (全买): {base_mean:.1f} bp")
    print(f"  - 基线策略开盘跳空 (Top10%): {strat_mean:.1f} bp")
    print(f"  - 创造 Alpha: {strat_mean - base_mean:.1f} bp")

if __name__ == "__main__":
    run_baselines()