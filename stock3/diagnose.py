# diagnose.py
import pandas as pd
import numpy as np

f = pd.read_parquet("D:/pyprj/cv2/data/processed/fact_clean.parquet")
f["trade_date"] = pd.to_datetime(f["trade_date"])

print("=" * 70)
print("P0: y_gap 极端值（|y_gap| > 22%）")
print("=" * 70)
ext = f[f["y_gap"].abs() > 0.22].copy()
print(f"极端样本数: {len(ext)}")
if len(ext):
    ext["gap_pct"] = (ext["y_gap"] * 100).round(2)
    cols = ["symbol", "trade_date", "y_gap", "gap_pct", "open", "pre_close", "listing_days", "is_st", "is_delist"]
    print(ext[cols].sort_values("y_gap").head(20).to_string(index=False))

print("\n" + "=" * 70)
print("P1: 停牌断档检测（被 label_valid=0 过滤掉的样本数）")
print("=" * 70)
print(f"总行数: {len(f)}")
print(f"有效样本 (label_valid=1): {int(f['label_valid'].sum())}")
print(f"无效样本 (label_valid=0): {int((f['label_valid']==0).sum())}")
# 看看因什么原因被过滤
invalid = f[f["label_valid"] == 0]
print(f" - 缺失下一日数据: {invalid['next_trade_date'].isna().sum()}")
print(f" - 次日不可交易/一字板: {((invalid['next_trade_date'].notna()) & ((invalid['tradable']==0) | (invalid['limit_locked']==1))).sum()}")
print(f" - 上市不足60天: {(invalid['listing_days'] < 60).sum()}")
print(f" - 退市/停牌跨越: {(invalid['is_delist']==1).sum()} (另外包含停牌跨越)")

print("\n" + "=" * 70)
print("P2: 按交易日的 y_gap 均值 / 方向 (仅查看 label_valid=1)")
print("=" * 70)
v = f[f["label_valid"] == 1]
daily = v.groupby("trade_date").agg(
    n=("y_gap", "size"),
    gap_mean_bp=("y_gap", lambda x: round(x.mean() * 1e4, 2)),
    gap_std_bp=("y_gap", lambda x: round(x.std() * 1e4, 2)),
    pct_up=("y_dir", lambda x: round(x.mean() * 100, 2)),
)
print(daily.to_string())