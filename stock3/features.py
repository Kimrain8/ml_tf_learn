# features.py
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from stock3.config import OUT_DIR

SEQ_COLS = [
    "pct_chg", "amplitude", "log_volume", "log_amount", "turnover",
    "volume_ratio", "main_inflow_ratio", "open_to_close", "high_to_close",
    "low_to_close", "gap_today",
]

TAB_COLS = [
    # 基础表格
    "chg_5d", "chg_10d", "chg_20d", "chg_60d", "chg_52w", "chg_ytd",
    "log_float_mv", "log_total_mv", "pb", "pe_ttm", "pe_dynamic", "pe_static",
    "main_inflow_ratio", "main_in_5d_ratio", "main_out_5d_ratio", "listing_days",
    # 个股锚点
    "prev_gap", "prev_ret", "prev_amp",
    # 市场层
    "mkt_pct", "prev_mkt_pct", "mkt_up_ratio", "prev_mkt_up_ratio",
    "mkt_amp", "mkt_limit_up", "mkt_limit_down",
    # 行业超额
    "ind_pct_excess",
]
MKT_COLS = ["mkt_pct", "prev_mkt_pct", "mkt_up_ratio", "prev_mkt_up_ratio",
            "mkt_amp", "mkt_limit_up", "mkt_limit_down"]

CAT_COLS = ["sw_l3_id"]
SEQ_LEN = 20

def _add_base_features(df):
    f = df.copy()
    f["log_volume"] = np.log1p(f["volume"])
    f["log_amount"] = np.log1p(f["amount"])
    f["log_float_mv"] = np.log1p(f["float_mv"])
    f["log_total_mv"] = np.log1p(f["total_mv"])
    f["open_to_close"] = (f["close"] - f["open"]) / f["open"].replace(0, np.nan)
    f["high_to_close"] = (f["high"] - f["close"]) / f["close"].replace(0, np.nan)
    f["low_to_close"] = (f["close"] - f["low"]) / f["close"].replace(0, np.nan)
    f["gap_today"] = (f["open"] - f["pre_close"]) / f["pre_close"].replace(0, np.nan)
    f["main_inflow_ratio"] = f["main_inflow"] / f["total_mv"].replace(0, np.nan)
    f["main_in_5d_ratio"] = f["main_in_5d"] / f["total_mv"].replace(0, np.nan)
    f["main_out_5d_ratio"] = f["main_out_5d"] / f["total_mv"].replace(0, np.nan)
    return f

def _add_anchor_features(df):
    f = df.sort_values(["symbol", "trade_date"]).copy()
    g = f.groupby("symbol")
    f["prev_gap"] = g["gap_today"].shift(1).fillna(0)
    f["prev_ret"] = g["pct_chg"].shift(1).fillna(0)
    f["prev_amp"] = g["amplitude"].shift(1).fillna(0)
    return f

def _add_market_features(df):
    f = df.copy()
    mkt = f.groupby("trade_date").agg(
        mkt_pct=("pct_chg", "mean"),
        mkt_up_ratio=("pct_chg", lambda x: (x > 0).mean()),
        mkt_amp=("amplitude", "mean"),
        mkt_limit_up=("limit_up", "sum"),
        mkt_limit_down=("limit_down", "sum"),
    ).reset_index()
    mkt["prev_mkt_pct"] = mkt["mkt_pct"].shift(1).fillna(0)
    mkt["prev_mkt_up_ratio"] = mkt["mkt_up_ratio"].shift(1).fillna(0)
    f = f.merge(mkt, on="trade_date", how="left")

    ind = f.groupby(["trade_date", "sw_l3_id"])["pct_chg"].mean().reset_index(name="ind_pct")
    f = f.merge(ind, on=["trade_date", "sw_l3_id"], how="left")
    f["ind_pct_excess"] = f["ind_pct"] - f["mkt_pct"]
    f = f.drop(columns=["ind_pct"])
    return f

def _rank_gauss_cross_section(df, cols):
    f = df.copy()
    def rg(x):
        if x.isna().all(): return x
        r = x.rank(method="average", na_option="keep")
        r = (r - 0.5) / r.notna().sum()
        return pd.Series(stats.norm.ppf(r), index=x.index)
    for c in cols:
        if c in f.columns:
            f[c] = f.groupby("trade_date")[c].transform(rg)
    return f

def _standardize(df):
    f = df.copy()
    # 时序特征全局 zscore（保持时间序列连续性）
    for c in SEQ_COLS:
        m, s = f[c].mean(), f[c].std()
        f[c] = (f[c] - m) / (s + 1e-8)
    # 市场层特征全局 zscore（同一天所有股票值相同，不能截面标准化）
    for c in MKT_COLS:
        if c in f.columns:
            m, s = f[c].mean(), f[c].std()
            f[c] = (f[c] - m) / (s + 1e-8)
    # 表格中除市场层外的特征按日 rank-gauss
    cross_cols = [c for c in TAB_COLS if c not in MKT_COLS]
    f = _rank_gauss_cross_section(f, cross_cols)
    return f

def _build_sequences(df):
    f = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    seq_data = []
    for col in SEQ_COLS:
        col_shifts = []
        for i in range(SEQ_LEN - 1, -1, -1):
            shifted = f.groupby("symbol")[col].shift(i)
            col_shifts.append(shifted.values)
        seq_data.append(np.column_stack(col_shifts))
    arr = np.stack(seq_data, axis=-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32)

def run_m3():
    print("=" * 60)
    print("M3 特征工程（全量生成，保留最新一天用于预测）")
    print("=" * 60)

    parquet_path = os.path.join(OUT_DIR, "fact_clean.parquet")
    df = pd.read_parquet(parquet_path)
    # ★★★ 新增：剔除北交所（4、8、920开头）以及退市整理期（如果未清理干净） ★★★
    df = df[~df["symbol"].str.startswith(("4", "8", "920"))].copy()
    print(f"过滤北交所后，待处理全量样本: {len(df)}")
    # print(f"待处理全量样本（含最新一天）: {len(df)}")

    df = _add_base_features(df)
    df = _add_anchor_features(df)
    df = _add_market_features(df)

    # 计算截面超额收益（作为模型优化的目标）
    df["y_gap_excess"] = df["y_gap"] - df.groupby("trade_date")["y_gap"].transform("mean")
    df["y_rank_pct"] = df.groupby("trade_date")["y_gap"].rank(pct=True)

    df = _standardize(df)

    print("构建时序张量...")
    X_seq = _build_sequences(df)

    print("构建表格张量...")
    X_tab = df[TAB_COLS].fillna(0).values.astype(np.float32)
    X_cat = df[CAT_COLS].fillna(0).values.astype(np.int64)

    # 提取标签并填充 NaN（最后一天没有下一日真实收益，会用 NaN 表示，这里填 0 占位）
    y_gap = df["y_gap"].fillna(0).values.astype(np.float32)
    y_gap_excess = df["y_gap_excess"].fillna(0).values.astype(np.float32)
    y_close_ret = df["y_close_ret"].fillna(0).values.astype(np.float32) if "y_close_ret" in df.columns else np.zeros(len(df))
    y_amp = df["y_amp"].fillna(0).values.astype(np.float32) if "y_amp" in df.columns else np.zeros(len(df))
    y_dir = df["y_dir"].fillna(0).values.astype(np.float32) if "y_dir" in df.columns else np.zeros(len(df))

    # meta 中必须包含 label_valid，供 train.py 识别真正能用于训练的样本
    meta = df[["symbol", "trade_date", "close", "open", "pre_close", "y_gap", "y_gap_excess", "label_valid"]].copy()

    out_dir = os.path.join(OUT_DIR, "m3_features")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_seq.npy"), X_seq)
    np.save(os.path.join(out_dir, "X_tab.npy"), X_tab)
    np.save(os.path.join(out_dir, "X_cat.npy"), X_cat)
    np.save(os.path.join(out_dir, "y_gap.npy"), y_gap)
    np.save(os.path.join(out_dir, "y_gap_excess.npy"), y_gap_excess)
    np.save(os.path.join(out_dir, "y_close_ret.npy"), y_close_ret)
    np.save(os.path.join(out_dir, "y_amp.npy"), y_amp)
    np.save(os.path.join(out_dir, "y_dir.npy"), y_dir)
    meta.to_parquet(os.path.join(out_dir, "meta.parquet"), index=False)

    print("\n" + "=" * 60)
    print(f"X_seq: {X_seq.shape}  X_tab: {X_tab.shape}  X_cat: {X_cat.shape}")
    print("=" * 60)

if __name__ == "__main__":
    run_m3()