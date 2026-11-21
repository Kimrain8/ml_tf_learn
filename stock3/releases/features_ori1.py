# features.py
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from stock3.config import OUT_DIR

# ==========================================
# 1. 特征列表配置（可按需增删）
# ==========================================
SEQ_COLS = [
    "pct_chg", "amplitude", "log_volume", "log_amount", "turnover", "volume_ratio",
    "main_inflow_ratio", "open_to_close", "high_to_close", "low_to_close", "gap_today"
]
TAB_COLS = [
    "chg_5d", "chg_10d", "chg_20d", "chg_60d", "chg_52w", "chg_ytd",
    "log_float_mv", "log_total_mv", "pb", "pe_ttm", "pe_dynamic", "pe_static",
    "main_inflow_ratio", "main_in_5d_ratio", "main_out_5d_ratio", "listing_days"
]
CAT_COLS = ["sw_l3_id"]

SEQ_LEN = 20  # 历史窗口长度

def _add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """衍生基础特征"""
    f = df.copy()
    # 对数化处理，防止极值影响
    f["log_volume"] = np.log1p(f["volume"])
    f["log_amount"] = np.log1p(f["amount"])
    f["log_float_mv"] = np.log1p(f["float_mv"])
    f["log_total_mv"] = np.log1p(f["total_mv"])
    
    # 相对位置/形态
    f["open_to_close"] = (f["close"] - f["open"]) / f["open"].replace(0, np.nan)
    f["high_to_close"] = (f["high"] - f["close"]) / f["close"].replace(0, np.nan)
    f["low_to_close"] = (f["close"] - f["low"]) / f["close"].replace(0, np.nan)
    f["gap_today"] = (f["open"] - f["pre_close"]) / f["pre_close"].replace(0, np.nan)
    
    # 资金流比率（除以总市值，横截面可比）
    f["main_inflow_ratio"] = f["main_inflow"] / f["total_mv"].replace(0, np.nan)
    f["main_in_5d_ratio"] = f["main_in_5d"] / f["total_mv"].replace(0, np.nan)
    f["main_out_5d_ratio"] = f["main_out_5d"] / f["total_mv"].replace(0, np.nan)
    
    return f

def _rank_gauss_cross_section(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """按日进行横截面正态秩变换，处理极值，将数据拉成标准正态分布"""
    f = df.copy()
    def rank_gauss(x):
        # 处理全NaN
        if x.isna().all(): return x
        # 排名并转化为0~1，防止为0或1
        r = x.rank(method="average", na_option="keep")
        r = (r - 0.5) / r.notna().sum()
        # 转化为正态分布分位数
        return pd.Series(stats.norm.ppf(r), index=x.index)

    for c in cols:
        if c in f.columns:
            f[c] = f.groupby("trade_date")[c].transform(rank_gauss)
    return f

def _build_sequences(df: pd.DataFrame) -> np.ndarray:
    """
    构造时序张量 [N, SEQ_LEN, F_seq]
    利用 pandas 的 shift 快速生成滑动窗口，比循环快得多
    """
    f = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    
    # 为每个特征生成 SEQ_LEN 个 shift
    seq_data = []
    for col in SEQ_COLS:
        col_shifts = []
        for i in range(SEQ_LEN - 1, -1, -1): # 从 T-19 到 T
            # 往过去 shift，所以 T-19 是最早的，shift(19)
            shifted = f.groupby("symbol")[col].shift(i)
            col_shifts.append(shifted.values)
        # 变成 [N, SEQ_LEN]
        seq_data.append(np.column_stack(col_shifts))
        
    # 拼成 [N, SEQ_LEN, F_seq]
    seq_array = np.stack(seq_data, axis=-1)
    # 时序里的 NaN（比如上市第一天不够20天）用 0 填充
    seq_array = np.nan_to_num(seq_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    return seq_array.astype(np.float32)

def run_m3():
    print("=" * 60)
    print("M3 开始特征工程")
    print("=" * 60)
    
    # 1. 读取清洗后的数据
    parquet_path = os.path.join(OUT_DIR, "fact_clean.parquet")
    df = pd.read_parquet(parquet_path)
    df = df[df["label_valid"] == 1].copy() # 只保留有效样本训练
    print(f"待处理有效样本: {len(df)}")
    
    # 2. 基础衍生特征
    df = _add_base_features(df)
    
    # 3. 横截面 Rank-Gauss 标准化（对时序和表格特征分别做）
    # 注意：时序特征需要按日做，这样时间轴上的绝对值才具有可比性
    print("执行按日横截面标准化...")
    df = _rank_gauss_cross_section(df, SEQ_COLS + TAB_COLS)
    
    # 4. 构建时序张量 [N, 20, F_seq]
    print("构建时序张量...")
    X_seq = _build_sequences(df)
    
    # 5. 构建表格特征张量 [N, F_tab]
    print("构建表格张量...")
    # 表格特征中如有 NaN，用 0 填充（因为已过 Rank-Gauss，0 代表均值）
    X_tab = df[TAB_COLS].fillna(0).values.astype(np.float32)
    
    # 6. 类别特征 [N]
    X_cat = df[CAT_COLS].fillna(0).values.astype(np.int64)
    
    # 7. 标签 [N]
    y_gap = df["y_gap"].values.astype(np.float32)
    y_close_ret = df["y_close_ret"].values.astype(np.float32)
    y_amp = df["y_amp"].fillna(0).values.astype(np.float32)
    y_dir = df["y_dir"].values.astype(np.float32)
    
    # 8. 元数据（用于后续评估和输出）
    meta = df[["symbol", "trade_date", "close", "open", "pre_close", "y_gap"]].copy()
    
    # 9. 保存到本地
    print("保存至本地...")
    out_dir = os.path.join(OUT_DIR, "m3_features")
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "X_seq.npy"), X_seq)
    np.save(os.path.join(out_dir, "X_tab.npy"), X_tab)
    np.save(os.path.join(out_dir, "X_cat.npy"), X_cat)
    
    np.save(os.path.join(out_dir, "y_gap.npy"), y_gap)
    np.save(os.path.join(out_dir, "y_close_ret.npy"), y_close_ret)
    np.save(os.path.join(out_dir, "y_amp.npy"), y_amp)
    np.save(os.path.join(out_dir, "y_dir.npy"), y_dir)
    
    meta.to_parquet(os.path.join(out_dir, "meta.parquet"), index=False)
    
    # 10. 报告
    print("\n" + "=" * 60)
    print("M3 完成！产出概览：")
    print(f"X_seq (时序): {X_seq.shape} 类型: {X_seq.dtype}")
    print(f"X_tab (表格): {X_tab.shape} 类型: {X_tab.dtype}")
    print(f"X_cat (类别): {X_cat.shape} 类型: {X_cat.dtype}")
    print(f"y_gap (标签): {y_gap.shape}")
    print(f"meta 行数: {len(meta)}")
    
    mem_seq = X_seq.nbytes / 1024**2
    mem_tab = X_tab.nbytes / 1024**2
    print(f"\n内存占用估计: 时序 {mem_seq:.1f} MB, 表格 {mem_tab:.1f} MB (非常适合 16G 内存)")
    print("=" * 60)

if __name__ == "__main__":
    run_m3()