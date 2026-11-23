# features_v2.py
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from config import OUT_DIR

# 个股时序特征 (11维)
SEQ_COLS = ["pct_chg", "amplitude", "log_volume", "log_amount", "turnover",
            "volume_ratio", "main_inflow_ratio", "open_to_close", 
            "high_to_close", "low_to_close", "gap_today"]
# 行业时序特征 (3维：行业日收益、行业日波动、行业换手)
IND_SEQ_COLS = ["ind_pct", "ind_amp", "ind_turnover"]
# 表格截面特征 (在原有基础上增加周期动态特征，28维)
TAB_COLS = ["chg_5d", "chg_10d", "chg_20d", "chg_60d", "chg_52w", "chg_ytd",
            "log_float_mv", "log_total_mv", "pb", "pe_ttm", "pe_dynamic", "pe_static",
            "main_inflow_ratio", "main_in_5d_ratio", "main_out_5d_ratio", "listing_days",
            "prev_gap", "prev_ret", "prev_amp",
            "mkt_pct", "prev_mkt_pct", "mkt_up_ratio", "prev_mkt_up_ratio",
            "mkt_amp", "mkt_limit_up", "mkt_limit_down", "ind_pct_excess",
            "ma20_distance"] # 新增：动态振幅（距离20日均线的偏离度）
# 7维
MKT_COLS = ["mkt_pct", "prev_mkt_pct", "mkt_up_ratio", "prev_mkt_up_ratio",
            "mkt_amp", "mkt_limit_up", "mkt_limit_down"]
# 行业ID
CAT_COLS = ["sw_l3_id"]
SEQ_LEN = 20

# 对量，金额，流通金额，总金额 做ln(1+x)处理；收盘对开盘增长率，最高价对收盘增长率，收盘对最低价增长率，
# 开盘对昨收盘增长率，主力净流入对总市值比例，5日净流入对总市值比例，5日净流出对总市值比例
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
    
    f = f.sort_values(["symbol", "trade_date"])
    ma20 = f.groupby("symbol")["close"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    f["ma20_distance"] = (f["close"] - ma20) / ma20.replace(0, np.nan)
    return f

def _add_anchor_features(df):
    f = df.sort_values(["symbol", "trade_date"]).copy()
    g = f.groupby("symbol")
    f["prev_gap"] = g["gap_today"].shift(1).fillna(0)
    f["prev_ret"] = g["pct_chg"].shift(1).fillna(0)
    f["prev_amp"] = g["amplitude"].shift(1).fillna(0)
    return f

def _add_market_and_industry_features(df):
    f = df.copy()
    mkt = f.groupby("trade_date").agg(
        mkt_pct=("pct_chg", "mean"),
        mkt_up_ratio=("pct_chg", lambda x: (x > 0).mean()),
        mkt_amp=("amplitude", "mean"),
        mkt_limit_up=("limit_up", "sum"),
        mkt_limit_down=("limit_down", "sum")).reset_index()
    mkt["prev_mkt_pct"] = mkt["mkt_pct"].shift(1).fillna(0)
    mkt["prev_mkt_up_ratio"] = mkt["mkt_up_ratio"].shift(1).fillna(0)
    f = f.merge(mkt, on="trade_date", how="left")
    
    # 将行业特征 merge 回 df
    ind = f.groupby(["trade_date", "sw_l3_id"]).agg(
        ind_pct=("pct_chg", "mean"),
        ind_amp=("amplitude", "mean"),
        ind_turnover=("turnover", "mean")
    ).reset_index()
    f = f.merge(ind, on=["trade_date", "sw_l3_id"], how="left")
    f["ind_pct_excess"] = f["ind_pct"] - f["mkt_pct"]
    
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
    for c in SEQ_COLS + IND_SEQ_COLS: # 把 IND_SEQ_COLS 也一起 Z-score
        m, s = f[c].mean(), f[c].std()
        f[c] = (f[c] - m) / (s + 1e-8)
    for c in MKT_COLS:
        if c in f.columns:
            m, s = f[c].mean(), f[c].std()
            f[c] = (f[c] - m) / (s + 1e-8)
    cross_cols = [c for c in TAB_COLS if c not in MKT_COLS]
    return _rank_gauss_cross_section(f, cross_cols)

def _build_sequences(df, cols, group_col="symbol"):
    f = df.sort_values([group_col, "trade_date"]).reset_index(drop=True)
    seq_data = []
    for col in cols:
        col_shifts = []
        for i in range(SEQ_LEN - 1, -1, -1):
            col_shifts.append(f.groupby(group_col)[col].shift(i).values)
        seq_data.append(np.column_stack(col_shifts))
    return np.nan_to_num(np.stack(seq_data, axis=-1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def run_m3_v2():
    print("=" * 60)
    print("M3 v2 特征工程 (修复行业张量维度对齐)")
    df = pd.read_parquet(os.path.join(OUT_DIR, "fact_clean.parquet"))
    df = df[~df["symbol"].str.startswith(("4", "8", "920"))].copy()
    print(f"待处理全量样本: {len(df)}")

    df = _add_base_features(df)
    df = _add_anchor_features(df)
    df = _add_market_and_industry_features(df)
    
    df["y_gap_excess"] = df["y_gap"] - df.groupby("trade_date")["y_gap"].transform("mean")
    std = df.groupby("trade_date")["y_gap_excess"].transform("std")
    df["y_gap_excess"] = df["y_gap_excess"].clip(-3 * std, 3 * std)
    
    df = _standardize(df)

    print("构建个股与行业时序张量...")
    X_seq = _build_sequences(df, SEQ_COLS, group_col="symbol")
    # ★ 直接对 df 里的 IND_SEQ_COLS 做序列化，得到和个股完全对齐的张量
    X_ind_seq = _build_sequences(df, IND_SEQ_COLS, group_col="symbol")
    
    print("构建表格张量...")
    X_tab = df[TAB_COLS].fillna(0).values.astype(np.float32)
    X_cat = df[CAT_COLS].fillna(0).values.astype(np.int64)

    meta = df[["symbol", "trade_date", "close", "open", "pre_close", "y_gap", "y_gap_excess", "label_valid"]].copy()

    out_dir = os.path.join(OUT_DIR, "m3_features_v2")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_seq.npy"), X_seq)
    np.save(os.path.join(out_dir, "X_ind_seq.npy"), X_ind_seq)
    np.save(os.path.join(out_dir, "X_tab.npy"), X_tab)
    np.save(os.path.join(out_dir, "X_cat.npy"), X_cat)
    np.save(os.path.join(out_dir, "y_gap_excess.npy"), df["y_gap_excess"].fillna(0).values.astype(np.float32))
    np.save(os.path.join(out_dir, "y_gap.npy"), df["y_gap"].fillna(0).values.astype(np.float32))
    meta.to_parquet(os.path.join(out_dir, "meta.parquet"), index=False)
    print(f"X_seq: {X_seq.shape}, X_ind_seq: {X_ind_seq.shape}, X_tab: {X_tab.shape}")

if __name__ == "__main__":
    run_m3_v2()