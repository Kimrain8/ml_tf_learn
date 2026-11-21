from IPython import embed
from utils.dbManager import MySQLHelper
# data_source.py
# ★ 唯一需要你手工编辑的文件。
# 每个函数返回一个 DataFrame，列名统一为下方标出的标准化列名。
#
# 用户原始列 → 标准化列 的映射：
#   股票代码         → symbol
#   交易日期         → trade_date
#   今日收盘价       → close
#   昨日收盘复权价   → pre_close
#   昨日收盘价       → prev_raw_close
#   （其他列名同原始中文名 → 见下方 alias 字典）
#
# 数据源2（如 tushare 风格）：
#   ts_code         → symbol（去后缀，如 300181.SZ → 300181）
#   trade_date      → trade_date
#   close_price     → close
#   pre_close       → pre_close
#   pre_close_ori   → prev_raw_close
#   pct_chg         → pct_chg
#
# 其余列（open/high/low/volume/...）保持原样映射。

import pandas as pd

# ---------- 通用列名别名（会自动 rename） ----------
COLUMN_ALIASES = {
    # 中文列名
    "股票代码": "symbol",
    "股票名称": "name",
    "交易日期": "trade_date",
    "今日收盘价": "close",
    "收盘价":   "close",
    "开盘价":   "open",
    "最高价":   "high",
    "最低价":   "low",
    "昨日收盘复权价": "pre_close",
    "昨日收盘价":     "prev_raw_close",
    "涨跌幅":   "pct_chg",
    "涨跌额":   "chg_amount",
    "近5日涨跌幅":  "chg_5d",
    "近10日涨跌幅": "chg_10d",
    "近20日涨跌幅": "chg_20d",
    "近60日涨跌幅": "chg_60d",
    "近52周涨跌幅": "chg_52w",
    "年初至今涨跌幅": "chg_ytd",
    "主力净流入":   "main_inflow",
    "近5日主力流入": "main_in_5d",
    "近5日主力流出": "main_out_5d",
    "换手率":   "turnover",
    "量比":     "volume_ratio",
    "价格振幅": "amplitude",
    "成交量":   "volume",
    "成交额":   "amount",
    "流通市值": "float_mv",
    "总市值":   "total_mv",
    "市净率":   "pb",
    "动态市盈率":   "pe_dynamic",
    "滚动市盈率":   "pe_ttm",
    "静态市盈率":   "pe_static",
    # tushare 风格
    "ts_code":     "symbol",
    "close_price": "close",
    "pre_close_ori": "prev_raw_close",
}

def _apply_alias(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})


# ================= 4 个加载函数 =================

def load_fact_daily() -> pd.DataFrame:
    """日频主表。"""
    # ★ 替换为你的查询
    # 方式1：从 CSV
    # df = pd.read_csv("data/raw/fact_daily.csv")
    # 方式2：从数据库
    # df = pd.read_sql("SELECT ... FROM fact_daily", conn)
    db = MySQLHelper(db='jhd')
    enddate = pd.Timestamp.now().date()
    startdate = enddate + pd.Timedelta(days=-70)
    sql = f"""select qh.symbol,qh.trade_date, tsh.close_price,qq.open_price,qq.high,qq.low
    ,tsh.pre_close,LAG(tsh.close_price, 1) OVER (PARTITION BY tsh.ts_code ORDER BY tsh.trade_date) AS prev_raw_close
    ,qh.pct_chg, qh.change_amount
    , qh.zdf_d5, qh.zdf_d10, qh.zdf_d20, qh.zdf_d60, qh.zdf_w52, qh.zdf_y
    , qh.zljlr,qh.zllr_d5,qh.zllc_d5
    , qh.hsl,qh.lb, qh.zf, qh.vol, qh.amount
    , qh.ltsz, qh.zsz,  qh.pn,qq.pe_dyna,qq.pe_ttm,qq.pe_static
    from qq_hsj AS qh left join qq_qt AS qq on qq.trade_date = qh.trade_date and qq.symbol = qh.symbol 
    join my_choice_stock AS mcs on mcs.symbol = qh.symbol
    join ts_stock_daily_his AS tsh on tsh.ts_code =  mcs.ts_code and qh.trade_date = tsh.trade_date
    where qh.trade_date between '{startdate.strftime("%Y-%m-%d")}' and '{enddate.strftime("%Y-%m-%d")}' """
    df = db.qry2df(sql=sql)
    # embed()
    df.rename(columns={"close_price":"close","open_price":"open",
               "change_amount":"chg_amount","zdf_d5":"chg_5d","zdf_d10":"chg_10d","zdf_d20":"chg_20d",
               "zdf_d60":"chg_60d","zdf_w52":"chg_52w","zdf_y":"chg_ytd",
               "zljlr":"main_inflow","zllr_d5":"main_in_5d","zllc_d5":"main_out_5d",
               "hsl":"turnover","lb":"volume_ratio","zf":"amplitude",
               "vol":"volume","ltsz":"float_mv","zsz":"total_mv",
               "pn":"pb","pe_dyna":"pe_dynamic"},inplace=True)
    min_date = df.groupby('symbol')['trade_date'].transform('min')
    df = df[df['trade_date'] > min_date].reset_index(drop=True)

    """df = pd.DataFrame(columns=[
        "symbol", "trade_date",
        "close", "open", "high", "low",
        "pre_close", "prev_raw_close",
        "pct_chg", "chg_amount",
        "chg_5d", "chg_10d", "chg_20d", "chg_60d", "chg_52w", "chg_ytd",
        "main_inflow", "main_in_5d", "main_out_5d",
        "turnover", "volume_ratio", "amplitude",
        "volume", "amount", "float_mv", "total_mv",
        "pb", "pe_dynamic", "pe_ttm", "pe_static",
    ])"""
    df = _apply_alias(df)
    # symbol 统一去后缀，纯数字
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    return df


def load_dim_stock() -> pd.DataFrame:
    """股票维度表。"""
    # ★ 替换为你的查询
    db = MySQLHelper(db='jhd')
    sql = f"""select asb.symbol,asb.name,asb.estab_date,asb.list_date,asb.exchange,
    asb.list_status,asb.camac_l1,asb.camac_l2,asb.sw_l1,asb.name_hy as sw_l2,asb.sw_l3 
    from ak_stock_basic AS asb"""
    df = db.qry2df(sql=sql)
    """df = pd.DataFrame(columns=[
        "symbol", "name", "estab_date", "list_date",
        "exchange", "list_status",
        "camac_l1", "camac_l2", "sw_l1", "sw_l2", "sw_l3",
    ])"""
    df = _apply_alias(df)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    return df


def load_ak_dividend() -> pd.DataFrame:
    """分红表。原样导出。"""
    # ★ 替换为你的查询
    db = MySQLHelper(db='jhd')
    sql = f"""select ad.symbol,ad.name,ad.end_date,ad.imp_ann_date,ad.imp_plan_desc,ad.stk_bo_rate,ad.stk_co_rate
    ,ad.cash_div_tax ,ad.cash_div,ad.record_date,ad.ex_date
    from ak_dividend AS ad 
    where ad.ex_date > '2026-01-01'"""
    df = db.qry2df(sql=sql)
    """df = pd.DataFrame(columns=[
        "symbol", "name", "end_date", "imp_ann_date", "imp_plan_desc",
        "stk_bo_rate", "stk_co_rate", "cash_div_tax", "cash_div",
        "record_date", "ex_date",
    ])"""
    df = _apply_alias(df)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    return df


def load_trade_calendar() -> pd.DataFrame:
    """交易日历。"""
    # ★ 替换为你的查询
    db = MySQLHelper(db='jhd')
    sql = f"""select ttd.exchange,ttd.trade_date,ttd.is_open,ttd.pretrade_date 
    from ts_trading_dates AS ttd where trade_date >= '2026-01-01'"""
    df = db.qry2df(sql=sql)
    """
    df = pd.DataFrame(columns=[
            "exchange", "trade_date", "is_open", "pretrade_date",
        ])
    """

    df = _apply_alias(df)
    return df

if __name__ == "__main__":
    df = load_fact_daily()