# schema.py
# 每个 df 的必需列, dtype, 缺失率上限

SCHEMA = {
    "fact_daily": {
        "required": [
            "symbol", "trade_date",
            "close", "open", "high", "low",
            "pre_close", "prev_raw_close",
            "pct_chg", "chg_amount",
            "chg_5d", "chg_10d", "chg_20d", "chg_60d", "chg_52w", "chg_ytd",
            "main_inflow", "main_in_5d", "main_out_5d",
            "turnover", "volume_ratio", "amplitude",
            "volume", "amount", "float_mv", "total_mv",
            "pb", "pe_dynamic", "pe_ttm", "pe_static",
        ],
        "dtype": {
            "symbol": "string", "trade_date": "datetime64[ns]",
            "close": "float64", "open": "float64",
            "high": "float64", "low": "float64",
            "pre_close": "float64", "prev_raw_close": "float64",
        },
        # 关键列缺失率上限
        "max_na_ratio": {
            "symbol": 0.0, "trade_date": 0.0,
            "close": 0.0, "open": 0.0, "pre_close": 0.0,
            "prev_raw_close": 0.0,
            "volume": 0.05, "amount": 0.05,
        },
    },
    "dim_stock": {
        "required": [
            "symbol", "name", "estab_date", "list_date",
            "exchange", "list_status",
            "camac_l1", "camac_l2", "sw_l1", "sw_l2", "sw_l3",
        ],
        "dtype": {
            "symbol": "string", "list_date": "datetime64[ns]",
            "sw_l3": "string", "list_status": "string",
        },
        "max_na_ratio": {"symbol": 0.0, "list_date": 0.02, "sw_l3": 0.05},
    },
    "ak_dividend": {
        "required": [
            "symbol", "ex_date",
            "cash_div_tax", "stk_bo_rate", "stk_co_rate",
        ],
        "dtype": {
            "symbol": "string", "ex_date": "datetime64[ns]",
            "cash_div_tax": "float64",
            "stk_bo_rate": "float64",
            "stk_co_rate": "float64",
        },
        # ex_date 允许 NULL（未实施方案），后面会丢
        "max_na_ratio": {"symbol": 0.0, "ex_date": 0.50},
    },
    "trade_calendar": {
        "required": ["exchange", "trade_date", "is_open", "pretrade_date"],
        "dtype": {"trade_date": "datetime64[ns]", "is_open": "int64"},
        "max_na_ratio": {"trade_date": 0.0, "is_open": 0.0},
    },
}