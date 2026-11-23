# predict_lgb.py
import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from config import OUT_DIR
from features_v2 import TAB_COLS

def get_latest_lgb_model():
    model_dir = os.path.join(OUT_DIR, "m3_features_v2")
    files = glob.glob(os.path.join(model_dir, "lgb_*.txt"))
    if not files: return None
    latest_file = sorted(files)[-1]
    print(f"自动加载最新 LGB 模型: {latest_file}")
    return latest_file

def run_predict_lgb(select_ratio=0.05, model_path=None):
    feat_dir = os.path.join(OUT_DIR, "m3_features_v2")
    X_tab = np.load(os.path.join(feat_dir, "X_tab.npy"))
    X_cat = np.load(os.path.join(feat_dir, "X_cat.npy"))
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))
    
    X_all = np.hstack([X_tab, X_cat])
    latest_date_str = str(meta["trade_date"].max()).split()[0]
    latest_date_fmt = latest_date_str.replace("-", "")
    
    mask = meta["trade_date"] == meta["trade_date"].max()
    
    if model_path is None:
        model_path = get_latest_lgb_model()
        if model_path is None:
            print("未找到任何历史模型，请先运行 train_lgb.py")
            return

    model = lgb.Booster(model_file=model_path)

    pred_excess = model.predict(X_all[mask])
    pred_excess = np.clip(pred_excess, -0.03, 0.03)

    result_df = meta[mask][["symbol", "close", "pre_close"]].copy()
    result_df["pred_excess_pct"] = pred_excess * 100
    result_df["today_pct"] = (result_df["close"] / result_df["pre_close"] - 1) * 100
    
    result_df = result_df.sort_values("pred_excess_pct", ascending=False).reset_index(drop=True)
    result_df["rank"] = result_df.index + 1
    
    n_select = max(1, int(len(result_df) * select_ratio))
    selected_df = result_df.head(n_select)

    print(f"\n===== LightGBM 选股 =====")
    print(selected_df[["rank", "symbol", "close", "today_pct", "pred_excess_pct"]].head(10).to_string(index=False))
    
    # ★ 输出 LGB 树模型的全局特征重要性
    print("\n" + "="*50)
    print("LightGBM 特征重要性 (全局):")
    importances = model.feature_importance(importance_type='gain')
    feature_names = TAB_COLS + ["sw_l3_id"]
    imp_df = pd.DataFrame({"Feature": feature_names, "Gain": importances}).sort_values("Gain", ascending=False)
    print(imp_df.head(5).to_string(index=False))
    imp_df.to_csv(os.path.join(OUT_DIR, f"lgb_feature_importance_{latest_date_fmt}.csv"), index=False, encoding="utf-8-sig")
    
    out_path = os.path.join(OUT_DIR, f"pred_lgb_{latest_date_fmt}.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] LGB 预测名单: {out_path}")

if __name__ == "__main__":
    run_predict_lgb(select_ratio=0.05)