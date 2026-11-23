# train_lgb.py
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from config import OUT_DIR
from features_v2 import TAB_COLS

def run_lgb():
    print("=" * 60)
    print("LightGBM 模型训练 (基于 V2 表格特征)")
    feat_dir = os.path.join(OUT_DIR, "m3_features_v2")
    
    # 1. 读取特征和标签
    X_tab = np.load(os.path.join(feat_dir, "X_tab.npy"))
    X_cat = np.load(os.path.join(feat_dir, "X_cat.npy"))
    y_gap_excess = np.load(os.path.join(feat_dir, "y_gap_excess.npy"))
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))

    # 2. 拼接连续特征和类别特征 (行业ID)
    # X_all = np.hstack([X_tab, X_cat])
    feature_names = TAB_COLS + ["sw_l3_id"]
    X_all = pd.DataFrame(np.hstack([X_tab, X_cat]), columns=feature_names)
    
    # 3. 严格时间切分
    valid_mask = meta["label_valid"] == 1
    meta_train = meta[valid_mask].copy()
    
    dates = sorted(meta_train["trade_date"].unique())
    n = len(dates)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    train_mask = meta_train["trade_date"].isin(dates[:train_end]).values
    val_mask = meta_train["trade_date"].isin(dates[train_end:val_end]).values
    test_mask = meta_train["trade_date"].isin(dates[val_end:]).values

    X_tr, y_tr = X_all[valid_mask][train_mask], y_gap_excess[valid_mask][train_mask]
    X_va, y_va = X_all[valid_mask][val_mask], y_gap_excess[valid_mask][val_mask]
    X_te, y_te = X_all[valid_mask][test_mask], y_gap_excess[valid_mask][test_mask]

    print(f"训练集: {len(X_tr)}, 验证集: {len(X_va)}, 测试集: {len(X_te)}")

    # 4. 训练 LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        num_leaves=31, min_child_samples=50, n_jobs=8, verbose=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(20, verbose=False)])

    # 5. 测试集 IC 评估
    pred_te = model.predict(X_te)
    test_meta = meta_train[test_mask].copy()
    test_meta["pred"] = pred_te
    test_meta["true_excess"] = y_te

    ic_list = []
    for d in sorted(test_meta["trade_date"].unique()):
        day_data = test_meta[test_meta["trade_date"] == d]
        if len(day_data) > 10:
            ic, _ = spearmanr(day_data["pred"], day_data["true_excess"])
            if not np.isnan(ic): ic_list.append(ic)
            
    print(f"LightGBM 测试集 Rank IC: {np.mean(ic_list):.4f}")
    
    # ★ 6. 按日保存 LGB 模型
    latest_date = str(meta["trade_date"].max()).split()[0].replace("-", "")
    model_path = os.path.join(OUT_DIR, "m3_features_v2", f"lgb_{latest_date}.txt")
    model.booster_.save_model(model_path)
    print(f"[保存] LightGBM 模型已保存至: {model_path}")

if __name__ == "__main__":
    run_lgb()
