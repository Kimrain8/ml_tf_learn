# config.py
# 全部口径与开关集中在此, 改这里即可

# ---- 单位换算：原单位 → 标准单位（乘数） ----
UNIT = {
    "volume":       100.0,   # 手 → 股
    "amount":       1e4,     # 万元 → 元
    "float_mv":     1e8,     # 亿元 → 元
    "total_mv":     1e8,     # 亿元 → 元
    "main_inflow":  1e4,     # 万元 → 元
    "main_in_5d":   1e4,     # 万元 → 元
    "main_out_5d":  1e4,     # 万元 → 元
    "pct_chg":      0.01,    # % → 小数
    "chg_5d":       0.01,
    "chg_10d":      0.01,
    "chg_20d":      0.01,
    "chg_60d":      0.01,
    "chg_52w":      0.01,
    "chg_ytd":      0.01,
    "turnover":     0.01,
    "amplitude":    0.01,
    "cash_div_tax": 0.1,     # 元/10股 → 元/股
    "cash_div":     0.1,     # 元/10股 → 元/股
    "stk_bo_rate":  0.1,     # 股/10股 → 股/股
    "stk_co_rate":  0.1,     # 股/10股 → 股/股
}

# ---- 涨跌停阈值 ----
LIMIT = {
    "main": 0.10,
    "gem":  0.20,   # 创业板 300/301
    "star": 0.20,   # 科创板 688
    "bse":  0.30,   # 北交所 4/8 开头
    "st":   0.05,
}

# ---- 除权样本处理开关 ----
# pre_close 已含除权调整，默认保留样本（keep）。
# 若想更保守，改为 "drop"，除权日样本从训练/评估中剔除。
EX_DIV_MODE = "keep"

# ---- 除权交叉验证容差（仅 warning，不阻塞） ----
EX_DIV_WARN_TOL = 0.02   # 2%

# ---- 输出路径 ----
OUT_DIR = "data/processed"