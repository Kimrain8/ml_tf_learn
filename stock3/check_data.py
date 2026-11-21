# check_data.py
# 入口: 加载 → 类型强制 → schema 校验 → 清洗 → 保存 parquet

import os
import pandas as pd
from stock3.data_source import (
    load_fact_daily, load_dim_stock,
    load_ak_dividend, load_trade_calendar,
)
from stock3.schema import SCHEMA
from stock3.normalize import coerce_types, full_pipeline
from stock3.config import OUT_DIR


def _check_schema(name: str, df: pd.DataFrame) -> bool:
    spec = SCHEMA[name]
    ok = True

    miss = [c for c in spec["required"] if c not in df.columns]
    if miss:
        print(f"[{name}] ✗ 缺列: {miss}")
        ok = False

    for col, want in spec.get("dtype", {}).items():
        if col not in df.columns:
            continue
        got = str(df[col].dtype)
        if want == "datetime64[ns]" and "datetime" not in got:
            print(f"[{name}] ! {col} dtype={got}, 期望 datetime")
        elif want not in ("datetime64[ns]",) and want not in got:
            print(f"[{name}] ! {col} dtype={got}, 期望 {want}")

    for col, max_na in spec.get("max_na_ratio", {}).items():
        if col not in df.columns:
            continue
        na = df[col].isna().mean()
        if na > max_na:
            print(f"[{name}] ✗ {col} 缺失率 {na:.2%} > {max_na:.2%}")
            ok = False

    if ok:
        print(f"[{name}] ✓ rows={len(df)}")
    return ok


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("M1 加载数据 + 类型强制")
    print("=" * 60)
    fact = coerce_types(load_fact_daily())
    dim  = coerce_types(load_dim_stock())
    div  = coerce_types(load_ak_dividend())
    cal  = coerce_types(load_trade_calendar())
    print(f"fact_daily rows={len(fact)}  dim_stock rows={len(dim)}")
    print(f"ak_dividend rows={len(div)}  trade_calendar rows={len(cal)}")

    print("=" * 60)
    print("M2-A schema 校验")
    print("=" * 60)
    all_ok = True
    all_ok &= _check_schema("fact_daily", fact)
    all_ok &= _check_schema("dim_stock", dim)
    all_ok &= _check_schema("ak_dividend", div)
    all_ok &= _check_schema("trade_calendar", cal)

    if not all_ok:
        print("\n[中止] schema 校验未通过，请先修复上方 ✗ 项。")
        return

    print("=" * 60)
    print("M2-B 单位换算 / 除权核查 / 标记 / 标签")
    print("=" * 60)
    fact_clean, ex_check = full_pipeline(fact, dim, div, cal)

    print("\n--- 除权交叉验证（前 10 条，按差异降序） ---")
    if not ex_check.empty:
        print(ex_check[["symbol", "trade_date", "pre_close",
                        "prev_raw_close", "expect_pre_close", "diff_pct"]]
              .sort_values("diff_pct", ascending=False).head(10).to_string(index=False))

    print("\n--- 清洗后主表概览 ---")
    print(f"总行数: {len(fact_clean)}")
    print(f"股票数: {fact_clean['symbol'].nunique()}")
    print(f"日期范围: {fact_clean['trade_date'].min().date()} ~ "
          f"{fact_clean['trade_date'].max().date()}")
    print(f"有效标签行数 (label_valid=1): {int(fact_clean['label_valid'].sum())}")

    print("\n--- 新增标记统计 ---")
    for c in ["tradable", "limit_up", "limit_down", "limit_locked",
              "is_ex_div", "is_st", "is_delist"]:
        if c in fact_clean.columns:
            print(f"  {c}=1: {int((fact_clean[c] == 1).sum())}")

    print("\n--- 标签分布 (label_valid=1) ---")
    v = fact_clean[fact_clean["label_valid"] == 1]
    if len(v):
        print(f"  y_gap 均值/标准差 (bp): "
              f"{v['y_gap'].mean()*1e4:.2f} / {v['y_gap'].std()*1e4:.2f}")
        print(f"  y_gap 最小值/最大值 (bp): "
              f"{v['y_gap'].min()*1e4:.2f} / {v['y_gap'].max()*1e4:.2f}")
        print(f"  方向分布: 涨 {(v['y_dir']==1).mean():.2%} / 跌 {(v['y_dir']==0).mean():.2%}")

    out_path = os.path.join(OUT_DIR, "fact_clean.parquet")
    fact_clean.to_parquet(out_path, index=False)
    print(f"\n[保存] {out_path}")

    ex_path = os.path.join(OUT_DIR, "ex_div_check.csv")
    if not ex_check.empty:
        ex_check.to_csv(ex_path, index=False, encoding="utf-8-sig")
        print(f"[保存] {ex_path}")


if __name__ == "__main__":
    main()