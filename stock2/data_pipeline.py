"""
数据源双通道：
  1) CSV：build_stock_dataset(cfg, csv_path=...)
  2) MySQL：build_stock_dataset(cfg, sql=..., db='jhd')
行业：申万三级 → 二级 → 一级 逐级回落
产品：主营业务 / 经营范围 → 关键词词典（business_classifier.py）
竞对：三级同行 → 二级同行 → 一级同行，按主营业务 TF-IDF 相似度取 top-N
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from business_classifier import (
    NUM_PRODUCT_CLASSES,
    PRODUCT_CLASS_NAMES,
    classify_main_business,
    tokenize_for_tfidf,
)


DEFAULT_STOCK_SQL = """
select qh.symbol,qh.name ,qh.trade_date, qh.zxj, qh.pct_chg, qh.change_amount, qh.hsl, qh.zljlr, qh.lb, qh.zf, qh.vol, qh.amount
, qh.ltsz, qh.zsz, qh.pn, qh.zdf_d5, qh.zdf_d10, qh.zdf_d20, qh.zdf_d60, qh.zdf_w52, qh.zdf_y
, qh.zllr, qh.zllc, qh.zllr_d5, qh.zllc_d5
,qq.high,qq.low,qq.open_price,qq.pe_dyna,qq.pe_ttm,qq.pe_static
,asb.sw_l1,asb.name_hy ,asb.sw_l3,asb.camac_l1,asb.camac_l2
,asb.bus_scope,asb.main_bus
from qq_hsj AS qh join qq_qt AS qq on qq.trade_date = qh.trade_date and qq.symbol = qh.symbol 
join ak_stock_basic AS asb on asb.symbol = qh.symbol 
where qq.trade_date >= '2026-09-30' and qq.trade_date <= '2026-10-28'
"""

DB_COLUMN_MAP = {
    'symbol': '股票代码', 'name': '股票名称', 'trade_date': '交易日期',
    'zxj': '最新价', 'pct_chg': '涨跌幅', 'change_amount': '涨跌额',
    'hsl': '换手率', 'zljlr': '主力净流入', 'lb': '量比', 'zf': '振幅',
    'vol': '成交量', 'amount': '成交额', 'ltsz': '流通市值', 'zsz': '总市值',
    'pe_ttm': '市盈率 (TTM)', 'pn': '市净率',
    'zdf_d5': '近5日涨跌幅', 'zdf_d10': '近10日涨跌幅',
    'zdf_d20': '近20日涨跌幅', 'zdf_d60': '近60日涨跌幅',
    'zdf_w52': '近52周涨跌幅', 'zdf_y': '年初至今涨跌幅',
    'zllr': '主力流入', 'zllc': '主力流出',
    'zllr_d5': '近 5 日主力流入', 'zllc_d5': '近 5 日主力流出',
    'high': '最高价', 'low': '最低价', 'open_price': '开盘价',
    'pe_dyna': '市盈率动', 'pe_static': '市盈率静',
    'sw_l1': '申万行业一级名称', 'name_hy': '申万行业二级名称',
    'sw_l3': '申万行业三级名称',
    'camac_l1': '中上协一级行业名称', 'camac_l2': '中上协二级行业名称',
    'bus_scope': '经营范围', 'main_bus': '主营业务',
}

FEATURE_SPECS = [
    ('涨跌幅', 'z'), ('振幅', 'z'), ('换手率', 'z'), ('量比', 'z'),
    ('成交量', 'log'), ('成交额', 'log'),
    ('主力净流入', 'log'), ('主力流入', 'log'), ('主力流出', 'log'),
    ('流通市值', 'log'), ('总市值', 'log'),
    ('市盈率 (TTM)', 'z'), ('市净率', 'z'),
    ('近5日涨跌幅', 'z'), ('近10日涨跌幅', 'z'), ('近20日涨跌幅', 'z'),
    ('最高价', 'rel'), ('最低价', 'rel'), ('开盘价', 'rel'), ('最新价', 'log'),
]
FEATURE_COLS = [s[0] for s in FEATURE_SPECS]

SW_COLS = {
    'sw1': '申万行业一级名称',
    'sw2': '申万行业二级名称',
    'sw3': '申万行业三级名称',
}


def _to_str(v):
    if v is None:
        return ''
    if isinstance(v, (bytes, bytearray)):
        for enc in ('utf-8', 'gbk', 'latin1'):
            try:
                return v.decode(enc)
            except Exception:
                continue
        return ''
    if isinstance(v, float) and np.isnan(v):
        return ''
    return str(v).strip()


def _fetch_from_db(sql, db='jhd', config_path=None):
    try:
        from utils.dbManager import MySQLHelper
    except ImportError as e:
        raise ImportError(f"无法导入 utils.dbManager.MySQLHelper：{e}")
    kwargs = {'db': db}
    if config_path is not None:
        kwargs['config_path'] = config_path
    helper = MySQLHelper(**kwargs)
    print(f"[db] 库={db}  SQL 前 200 字符：{sql.strip()[:200]}...")
    df = helper.qry2df(sql)
    print(f"[db] 返回 {len(df)} 行 × {len(df.columns)} 列")
    return _rename_db_columns(df)


def _rename_db_columns(df):
    if df is None or df.empty:
        return df
    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated()].tolist()
        print(f"[info] 忽略 SQL 中重复返回的列：{dup}")
        df = df.loc[:, ~df.columns.duplicated()].copy()
    missing = [c for c in DB_COLUMN_MAP if c not in df.columns]
    if missing:
        print(f"[warn] SQL 结果中缺少以下列：{missing}")
    df = df.rename(columns=DB_COLUMN_MAP)
    for col in ('主营业务', '经营范围'):
        if col in df.columns:
            df[col] = df[col].apply(_to_str)
    return df


def _load_csv(csv_path):
    df = pd.read_csv(csv_path)
    if '申万行业一级名称.1' in df.columns:
        df = df.rename(columns={'申万行业一级名称.1': '申万行业二级名称'})
    dup_suffix = [c for c in df.columns if c.endswith('.1')]
    if dup_suffix:
        df = df.drop(columns=dup_suffix)
    for col in ('主营业务', '经营范围'):
        if col in df.columns:
            df[col] = df[col].apply(_to_str)
    return df


def _apply_transform(g, specs):
    T = len(g)
    X = np.zeros((T, len(specs)), dtype=np.float64)
    close = g['最新价'].astype(float).to_numpy(dtype=np.float64)
    for j, (col, t) in enumerate(specs):
        v = g[col].astype(float).to_numpy(dtype=np.float64)
        if t == 'log':
            v = np.sign(v) * np.log1p(np.abs(v))
        elif t == 'rel':
            v = v / (close + 1e-8)
        X[:, j] = v
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32)


def _pick_industry_name(sw3, sw2, sw1):
    for v in (sw3, sw2, sw1):
        if isinstance(v, str):
            v = v.strip()
            if v and v not in ('nan', 'None', '未知', ''):
                return v
    return '未知'


def _build_product_labels(df_stocks):
    labels = np.zeros(len(df_stocks), dtype=np.int64)
    sids = df_stocks['_stock_id'].to_numpy()
    main_bizs = df_stocks['main_biz'].tolist()
    scopes = df_stocks['scope'].tolist()
    for sid, mb, sc in zip(sids, main_bizs, scopes):
        labels[sid] = classify_main_business(mb, sc)
    return labels


def _build_competitor_map(df_stocks, max_comp, seed, batch_size=2000):
    n = len(df_stocks)
    texts = [
        tokenize_for_tfidf(f"{row.main_biz} {row.scope}")
        for row in df_stocks.itertuples(index=False)
    ]
    vec = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=8000)
    mat = vec.fit_transform(texts)

    sw1_list = df_stocks['sw1'].astype(str).tolist()
    sw2_list = df_stocks['sw2'].astype(str).tolist()
    sw3_list = df_stocks['sw3'].astype(str).tolist()

    sw1_groups = defaultdict(list)
    sw2_groups = defaultdict(list)
    sw3_groups = defaultdict(list)
    for i in range(n):
        sw1_groups[sw1_list[i]].append(i)
        sw2_groups[sw2_list[i]].append(i)
        sw3_groups[sw3_list[i]].append(i)

    comp_map = {}
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        sim_block = cosine_similarity(mat[batch_start:batch_end], mat)
        for i in range(batch_start, batch_end):
            row = i - batch_start
            seen = {i}
            cand = []
            for j in sw3_groups[sw3_list[i]]:
                if j not in seen:
                    seen.add(j); cand.append(j)
            if len(cand) < max_comp:
                for j in sw2_groups[sw2_list[i]]:
                    if j not in seen:
                        seen.add(j); cand.append(j)
            if len(cand) < max_comp:
                for j in sw1_groups[sw1_list[i]]:
                    if j not in seen:
                        seen.add(j); cand.append(j)
            if not cand:
                comp_map[i] = []
                continue
            cand_arr = np.array(cand, dtype=np.int64)
            sims = sim_block[row, cand_arr]
            order = np.argsort(-sims)
            comp_map[i] = cand_arr[order[:max_comp]].tolist()
    return comp_map


def build_stock_dataset(
    cfg, csv_path=None, df=None, sql=None, db='jhd', db_config_path=None,
    val_days=0, seed=42, main_biz_col='主营业务', biz_scope_col='经营范围',
):
    rng = np.random.default_rng(seed)

    if df is None:
        if sql is not None:
            df = _fetch_from_db(sql, db=db, config_path=db_config_path)
        elif csv_path is not None:
            df = _load_csv(csv_path)
        else:
            raise ValueError("必须提供 df / sql / csv_path 三者之一")

    df = df.copy()
    df['交易日期'] = pd.to_datetime(df['交易日期'], errors='coerce')
    df = df.dropna(subset=['交易日期'])
    df = df.sort_values(['股票代码', '交易日期']).reset_index(drop=True)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"数据中缺少特征列：{missing}")
    for key, col in SW_COLS.items():
        if col not in df.columns:
            print(f"[warn] 缺少申万列 {col}（层级 {key}），该层级视为空")
            df[col] = '未知'

    df[FEATURE_COLS] = (
        df.groupby('股票代码')[FEATURE_COLS]
          .transform(lambda s: pd.to_numeric(s, errors='coerce').ffill().bfill())
    )
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("清洗后数据为空")

    stock_codes = sorted(df['股票代码'].astype(str).unique().tolist())
    stock_to_id = {c: i for i, c in enumerate(stock_codes)}
    df['股票代码'] = df['股票代码'].astype(str)
    df['_stock_id'] = df['股票代码'].map(stock_to_id).astype(np.int64)

    df['_sw1'] = df[SW_COLS['sw1']].astype(str).str.strip()
    df['_sw2'] = df[SW_COLS['sw2']].astype(str).str.strip()
    df['_sw3'] = df[SW_COLS['sw3']].astype(str).str.strip()
    df['_industry_name'] = [
        _pick_industry_name(a, b, c)
        for a, b, c in zip(df['_sw3'], df['_sw2'], df['_sw1'])
    ]

    industry_names = sorted(df['_industry_name'].unique().tolist())
    ind_name_to_id = {name: i for i, name in enumerate(industry_names)}
    ind_id_to_name = {i: name for name, i in ind_name_to_id.items()}
    df['_industry_id'] = df['_industry_name'].map(ind_name_to_id).astype(np.int64)
    num_industries = len(industry_names)
    print(f"[info] 主行业 ID 映射：共 {num_industries} 个")
    for i in range(min(5, num_industries)):
        print(f"    {i:>3d} → {ind_id_to_name[i]}")

    first_per_stock = (
        df.sort_values('交易日期').groupby('_stock_id', as_index=False).first()
    )
    first_per_stock['main_biz'] = (
        first_per_stock[main_biz_col].apply(_to_str)
        if main_biz_col in first_per_stock.columns else ''
    )
    first_per_stock['scope'] = (
        first_per_stock[biz_scope_col].apply(_to_str)
        if biz_scope_col in first_per_stock.columns else ''
    )
    first_per_stock['sw1'] = first_per_stock['_sw1']
    first_per_stock['sw2'] = first_per_stock['_sw2']
    first_per_stock['sw3'] = first_per_stock['_sw3']

    product_labels = _build_product_labels(first_per_stock)
    print(f"[info] 产品类别数={NUM_PRODUCT_CLASSES}（含“其他”）")
    uniq, cnt = np.unique(product_labels, return_counts=True)
    print("[info] 产品类别分布 top10：")
    for k in np.argsort(-cnt)[:10]:
        print(f"    {PRODUCT_CLASS_NAMES[uniq[k]]:<10s} {cnt[k]:>5d} 只")

    comp_map = _build_competitor_map(first_per_stock, cfg.max_competitors, seed)
    n_hist = defaultdict(int)
    for _, lst in comp_map.items():
        n_hist[len(lst)] += 1
    print("[info] 每只股票竞争对手数量分布（含 0）：")
    for k in sorted(n_hist):
        print(f"    {k} 个：{n_hist[k]} 只股票")

    X_list, ret_list, sid_list, end_date_list = [], [], [], []
    H, F = cfg.history_days, cfg.future_days
    for code, g in df.groupby('股票代码', sort=False):
        g = g.sort_values('交易日期')
        if len(g) < H + F:
            continue
        X = _apply_transform(g, FEATURE_SPECS)
        close = g['最新价'].astype(float).to_numpy(dtype=np.float64)
        dates = g['交易日期'].to_numpy()
        sid = stock_to_id[code]
        last_i = len(X) - H - F
        for i in range(last_i + 1):
            c_now = close[i + H - 1]
            c_fut = close[i + H - 1 + F]
            if c_now <= 0 or c_fut <= 0:
                continue
            X_list.append(X[i:i + H])
            ret_list.append(np.log(c_fut / c_now))
            sid_list.append(sid)
            end_date_list.append(dates[i + H - 1 + F])

    if not X_list:
        raise RuntimeError("样本数为 0，检查 history_days / future_days 与单股长度")

    X_arr = np.stack(X_list).astype(np.float32)
    ret_arr = np.asarray(ret_list, dtype=np.float32)
    sid_arr = np.asarray(sid_list, dtype=np.int64)
    end_dates = np.array(end_date_list)
    N = len(X_arr)

    flat = X_arr.reshape(-1, X_arr.shape[-1])
    mean = flat.mean(0).astype(np.float32)
    std = (flat.std(0) + 1e-8).astype(np.float32)
    X_arr = (X_arr - mean) / std

    ind_per_stock = dict(zip(first_per_stock['_stock_id'], first_per_stock['_industry_id']))
    industry_arr = np.array([ind_per_stock[s] for s in sid_arr], dtype=np.int64)
    product_arr = product_labels[sid_arr]

    comp_arr = np.full((N, cfg.max_competitors), -1, dtype=np.int64)
    for n, sid in enumerate(sid_arr):
        top = comp_map.get(int(sid), [])
        if top:
            k = min(cfg.max_competitors, len(top))
            comp_arr[n, :k] = top[:k]

    cfg.stock_feat_dim = X_arr.shape[-1]
    cfg.num_industries = num_industries
    cfg.num_products = NUM_PRODUCT_CLASSES
    cfg.total_stocks = len(stock_codes)
    cfg._finalize()

    # ---------- 切分：val_days=0 全部训练 ----------
    all_dates = np.unique(df['交易日期'].to_numpy())
    all_dates.sort()
    n_days_total = len(all_dates)
    print(f"[split] 全市场交易日数: {n_days_total} "
          f"({pd.Timestamp(all_dates[0]).date()} ~ {pd.Timestamp(all_dates[-1]).date()})")

    if val_days == 0:
        print("[split] val_days=0 → 全部样本用于训练，无验证集")
        train_idx = np.arange(N)
        val_idx = np.array([], dtype=np.int64)
        val_start_date = None
        train_range = (
            (pd.Timestamp(end_dates.min()).date(),
             pd.Timestamp(end_dates.max()).date()) if N > 0 else None
        )
        val_range = None
    else:
        if n_days_total <= val_days:
            raise RuntimeError(f"交易日数({n_days_total}) <= 验证天数({val_days})")
        val_start_date = all_dates[-val_days]
        print(f"[split] 验证起始日: {pd.Timestamp(val_start_date).date()} "
              f"（最后 {val_days} 个交易日）")
        is_val = end_dates >= val_start_date
        train_idx = np.where(~is_val)[0]
        val_idx = np.where(is_val)[0]

        def _fmt(d): return pd.Timestamp(d).date() if d is not None else None
        train_range = (
            (_fmt(end_dates[train_idx].min()), _fmt(end_dates[train_idx].max()))
            if len(train_idx) > 0 else None
        )
        val_range = (
            (_fmt(end_dates[val_idx].min()), _fmt(end_dates[val_idx].max()))
            if len(val_idx) > 0 else None
        )
        print(f"[split] 训练集: {len(train_idx):>7d} 样本  目标日范围: {train_range}")
        print(f"[split] 验证集: {len(val_idx):>7d} 样本  目标日范围: {val_range}")
        if len(train_idx) == 0:
            raise RuntimeError(
                f"❌ 训练集样本数为 0！\n"
                f"  全市场交易日数: {n_days_total}\n"
                f"  最后 {val_days} 天作验证，训练可用 {n_days_total - val_days} 天\n"
                f"  每样本需要 history_days({cfg.history_days}) + "
                f"future_days({cfg.future_days}) = "
                f"{cfg.history_days + cfg.future_days} 天\n"
                f"  解决: --val_days 0，或放宽 SQL 日期范围"
            )

    def _slice(idx):
        return {
            'features': X_arr[idx],
            'industry_id': industry_arr[idx],
            'product_id': product_arr[idx],
            'competitor_ids': comp_arr[idx],
            'future_return': ret_arr[idx],
        }

    meta = {
        'stock_to_id': stock_to_id,
        'industry_name_to_id': ind_name_to_id,
        'industry_id_to_name': ind_id_to_name,
        'product_class_names': PRODUCT_CLASS_NAMES,
        'mean': mean,
        'std': std,
        'feature_cols': FEATURE_COLS,
        'n_total': N,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'val_days': val_days,
        'val_start_date': str(val_start_date) if val_start_date is not None else None,
        'train_target_range': train_range,
        'val_target_range': val_range,
        'competitor_map': comp_map,   # {stock_id: [comp_id, ...]}
    }

    train_data = _slice(train_idx)
    val_data = _slice(val_idx) if len(val_idx) > 0 else None
    return train_data, val_data, meta