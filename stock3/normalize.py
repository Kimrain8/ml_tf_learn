# normalize.py
# 单位换算, 类型强制, 除权交叉验证, 标记生成, 标签构造

import numpy as np
import pandas as pd
from stock3.config import UNIT, LIMIT, EX_DIV_MODE, EX_DIV_WARN_TOL

# ---------- 0. 类型列表 ----------
NUMERIC_COLS = [
    "close", "open", "high", "low", "pre_close", "prev_raw_close",
    "pct_chg", "chg_amount",
    "chg_5d", "chg_10d", "chg_20d", "chg_60d", "chg_52w", "chg_ytd",
    "main_inflow", "main_in_5d", "main_out_5d",
    "turnover", "volume_ratio", "amplitude",
    "volume", "amount", "float_mv", "total_mv",
    "pb", "pe_dynamic", "pe_ttm", "pe_static",
    "cash_div_tax", "cash_div", "stk_bo_rate", "stk_co_rate",
]

DATETIME_COLS = [
    "trade_date", "ex_date", "list_date", "estab_date",
    "pretrade_date", "imp_ann_date", "record_date", "end_date",
]

STRING_COLS = [
    "symbol", "name", "exchange", "list_status",
    "camac_l1", "camac_l2", "sw_l1", "sw_l2", "sw_l3",
]

INT_COLS = ["is_open"]


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in DATETIME_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in STRING_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")
    for c in INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    return df


# ---------- 1. 单位换算 ----------
def apply_units(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, factor in UNIT.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * factor
    return df


# ---------- 2. 用 pre_close 强制重算 pct_chg ----------
def recompute_pct_chg(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f["pct_chg_src"] = f["pct_chg"] if "pct_chg" in f.columns else np.nan
    f["pct_chg_calc"] = (f["close"] - f["pre_close"]) / f["pre_close"].replace(0, np.nan)
    f["pct_chg"] = f["pct_chg_calc"]
    return f


# ---------- 3. 除权交叉验证 ----------
def cross_check_ex_div(f: pd.DataFrame, div: pd.DataFrame) -> pd.DataFrame:
    if div.empty:
        print("[cross_check_ex_div] 分红表为空，跳过")
        return pd.DataFrame()

    d = div.copy()
    d = d.dropna(subset=["ex_date"]).drop_duplicates(["symbol", "ex_date"])
    d["cash_per_share"] = d["cash_div_tax"].fillna(0)
    d["ratio_per_share"] = d["stk_bo_rate"].fillna(0) + d["stk_co_rate"].fillna(0)

    m = f.merge(
        d[["symbol", "ex_date", "cash_per_share", "ratio_per_share"]],
        how="inner",
        left_on=["symbol", "trade_date"],
        right_on=["symbol", "ex_date"],
    )
    if m.empty:
        print("[cross_check_ex_div] 无除权日样本")
        return m

    m["expect_pre_close"] = (m["prev_raw_close"] - m["cash_per_share"]) / (1 + m["ratio_per_share"])
    m["diff_pct"] = (m["pre_close"] - m["expect_pre_close"]).abs() / m["expect_pre_close"].replace(0, np.nan)

    bad = m[m["diff_pct"] > EX_DIV_WARN_TOL]
    print(f"[cross_check_ex_div] 除权样本 {len(m)} 个，差异超阈值 {len(bad)} 个")
    if len(bad):
        print(bad[["symbol", "trade_date", "pre_close", "prev_raw_close", "expect_pre_close", "diff_pct"]]
              .sort_values("diff_pct", ascending=False).head(10))
    return m


# ---------- 4. 维度附加 ----------
def _attach_dim(f: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame:
    d = dim.copy()
    d["is_st"] = d["list_status"].astype(str).str.upper().isin(["ST", "*ST"]).astype(int)
    d["is_delist"] = d["list_status"].astype(str).str.contains("退市", na=False).astype(int)

    f = f.merge(
        d[["symbol", "list_date", "estab_date", "sw_l3", "is_st", "is_delist"]],
        on="symbol", how="left",
    )
    f["is_st"] = f["is_st"].fillna(0).astype(int)
    f["is_delist"] = f["is_delist"].fillna(0).astype(int)

    names = sorted(f["sw_l3"].dropna().unique().tolist())
    sw_map = {n: i + 1 for i, n in enumerate(names)}
    sw_map["__UNK__"] = 0
    f["sw_l3_id"] = f["sw_l3"].map(sw_map).fillna(0).astype(int)

    f["listing_days"] = (f["trade_date"] - f["list_date"]).dt.days.clip(lower=0)
    f = f.drop(columns=["list_date", "estab_date"])
    return f


# ---------- 5. 市场标记 ----------
def _limit_of(symbol: str, is_st: int) -> float:
    if is_st: return LIMIT["st"]
    s = str(symbol).zfill(6)
    if s.startswith(("300", "301")): return LIMIT["gem"]
    if s.startswith("688"): return LIMIT["star"]
    if s.startswith(("4", "8")): return LIMIT["bse"]
    return LIMIT["main"]


def build_market_flags(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f["tradable"] = ((f["volume"].fillna(0) > 0) & (f["amount"].fillna(0) > 0)).astype(int)
    f["limit_pct"] = [_limit_of(s, st) for s, st in zip(f["symbol"], f["is_st"])]
    f["limit_up"]   = (f["pct_chg"] >=  f["limit_pct"] - 0.005).astype(int)
    f["limit_down"] = (f["pct_chg"] <= -f["limit_pct"] + 0.005).astype(int)
    f["limit_locked"] = ((f["high"] == f["low"]) & ((f["limit_up"] == 1) | (f["limit_down"] == 1))).astype(int)
    return f


# ---------- 6. 标记除权样本 ----------
def mark_ex_div(f: pd.DataFrame, div: pd.DataFrame) -> pd.DataFrame:
    if div.empty:
        f = f.copy()
        f["is_ex_div"] = 0
        f["ex_div_drop"] = 0
        return f
    d = div.dropna(subset=["ex_date"]).drop_duplicates(["symbol", "ex_date"]).copy()
    d["is_ex_div"] = 1
    f = f.merge(d[["symbol", "ex_date", "is_ex_div"]], how="left",
                left_on=["symbol", "trade_date"], right_on=["symbol", "ex_date"])
    f["is_ex_div"] = f["is_ex_div"].fillna(0).astype(int)
    f = f.drop(columns=["ex_date"])
    f["ex_div_drop"] = ((EX_DIV_MODE == "drop") & (f["is_ex_div"] == 1)).astype(int)
    return f


# ---------- 7. 构造标签（含连续交易日与异常过滤） ----------
def build_labels(f: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    f = f.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    g = f.groupby("symbol", sort=False)

    f["next_trade_date"]   = g["trade_date"].shift(-1)
    f["next_open"]         = g["open"].shift(-1)
    f["next_pre_close"]    = g["pre_close"].shift(-1)
    f["next_close"]        = g["close"].shift(-1)
    f["next_amplitude"]    = g["amplitude"].shift(-1)
    f["next_limit_locked"] = g["limit_locked"].shift(-1)
    f["next_tradable"]     = g["tradable"].shift(-1)

    f["y_gap"]       = np.log(f["next_open"] / f["next_pre_close"])
    f["y_close_ret"] = np.log(f["next_close"] / f["close"])
    f["y_amp"]       = f["next_amplitude"]
    f["y_dir"]       = (f["y_gap"] > 0).astype(int)

    # 判断是否严格相邻交易日（防止停牌跨越）
    cal_seq = cal.sort_values("trade_date").reset_index(drop=True)
    cal_seq["seq"] = cal_seq.index
    seq_map = dict(zip(cal_seq["trade_date"], cal_seq["seq"]))
    
    f["seq_t"] = f["trade_date"].map(seq_map)
    f["seq_next"] = f["next_trade_date"].map(seq_map)
    f["is_consecutive"] = (f["seq_next"] - f["seq_t"]) == 1

    # 综合判断有效标签
    f["label_valid"] = (
        f["next_trade_date"].notna()
        & (f["next_tradable"] == 1)
        & (f["next_limit_locked"] == 0)
        & (f.get("ex_div_drop", 0) == 0)
        & (f["is_consecutive"] == True)        # 中间不能停牌
        & (f["listing_days"] >= 60)            # 剔除次新股（上市不足60自然日）
        & (f["y_gap"].abs() <= 0.22)           # 剔除大于22%的异常跳空（防止脏数据或首日）
        & (f["is_delist"] == 0)                # 剔除退市整理期
    ).astype(int)

    # 清理中间列，防止后续做特征时发生未来函数
    f = f.drop(columns=["next_open", "next_pre_close", "next_close", 
                        "next_amplitude", "seq_t", "seq_next", "is_consecutive"])
    return f


# ---------- 8. 一键 pipeline ----------
def full_pipeline(fact: pd.DataFrame, dim: pd.DataFrame,
                  div: pd.DataFrame, cal: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    fact = coerce_types(fact)
    dim  = coerce_types(dim)
    div  = coerce_types(div)
    cal  = coerce_types(cal)

    fact = apply_units(fact)
    div_u = apply_units(div)

    fact = fact.dropna(subset=["symbol", "trade_date", "close", "open", "pre_close"])
    fact = recompute_pct_chg(fact)
    ex_check = cross_check_ex_div(fact, div_u)

    fact = _attach_dim(fact, dim)
    fact = build_market_flags(fact)
    fact = mark_ex_div(fact, div_u)
    fact = build_labels(fact, cal)  # 传入 cal

    return fact, ex_check