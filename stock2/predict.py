"""
用训练好的模型做批量推理（不训练）。
- 行业 ID：查 meta['industry_name_to_id']，未见过 → 0
- 产品 ID：classify_main_business() 实时算（确定性，不需要训练集）
- 竞对：查 meta['competitor_map']，新股票 → -1
- 标准化：用 meta['mean'] / meta['std']
"""
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from model import StockTransformer
from data_pipeline import (
    FEATURE_SPECS, SW_COLS, DB_COLUMN_MAP, DEFAULT_STOCK_SQL,
    _apply_transform, _pick_industry_name, _to_str,
)
from business_classifier import classify_main_business


def load_checkpoint(path, device):
    # ckpt = torch.load(path, map_location=device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ckpt['cfg'])
    cfg.device = device
    cfg._finalize()
    model = StockTransformer(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, cfg, ckpt['meta']


def fetch_df(args):
    if args.csv:
        df = pd.read_csv(args.csv)
        if '申万行业一级名称.1' in df.columns:
            df = df.rename(columns={'申万行业一级名称.1': '申万行业二级名称'})
        dup = [c for c in df.columns if c.endswith('.1')]
        if dup:
            df = df.drop(columns=dup)
    else:
        from utils.dbManager import MySQLHelper
        sql = args.sql
        if sql is None and args.sql_file:
            with open(args.sql_file, 'r', encoding='utf-8') as f:
                sql = f.read()
        if sql is None:
            sql = DEFAULT_STOCK_SQL
            print("[data] 使用内置 DEFAULT_STOCK_SQL（如需换时间段请传 --sql）")

        helper_kwargs = {'db': args.db}
        if args.db_config is not None:
            helper_kwargs['config_path'] = args.db_config
        helper = MySQLHelper(**helper_kwargs)
        df = helper.qry2df(sql)
        
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()
        df = df.rename(columns=DB_COLUMN_MAP)

    for col in ('主营业务', '经营范围'):
        if col in df.columns:
            df[col] = df[col].apply(_to_str)
    df['交易日期'] = pd.to_datetime(df['交易日期'], errors='coerce')
    df = df.dropna(subset=['交易日期'])
    df = df.sort_values(['股票代码', '交易日期']).reset_index(drop=True)
    return df


def build_inference_samples(df, cfg, meta):
    stock_to_id = meta['stock_to_id']
    ind_name_to_id = meta['industry_name_to_id']
    competitor_map = meta.get('competitor_map', {})

    for key, col in SW_COLS.items():
        if col not in df.columns:
            print(f"[warn] 缺少 {col}，该层级视为空")
            df[col] = '未知'

    df['_sw1'] = df[SW_COLS['sw1']].astype(str).str.strip()
    df['_sw2'] = df[SW_COLS['sw2']].astype(str).str.strip()
    df['_sw3'] = df[SW_COLS['sw3']].astype(str).str.strip()
    df['_industry_name'] = [
        _pick_industry_name(a, b, c)
        for a, b, c in zip(df['_sw3'], df['_sw2'], df['_sw1'])
    ]

    known = set(ind_name_to_id.keys())
    unknown = sorted(set(df['_industry_name'].unique()) - known)
    if unknown:
        print(f"[warn] {len(unknown)} 个行业未在训练集见过，回退到 id=0：")
        for n in unknown[:10]:
            print(f"    '{n}'")

    first_per_stock = (
        df.sort_values('交易日期').groupby('股票代码', as_index=False).first()
    )
    main_biz_map = dict(zip(
        first_per_stock['股票代码'].astype(str),
        first_per_stock['主营业务'].apply(_to_str) if '主营业务' in first_per_stock.columns else [''] * len(first_per_stock),
    ))
    scope_map = dict(zip(
        first_per_stock['股票代码'].astype(str),
        first_per_stock['经营范围'].apply(_to_str) if '经营范围' in first_per_stock.columns else [''] * len(first_per_stock),
    ))
    ind_map = dict(zip(first_per_stock['股票代码'].astype(str), first_per_stock['_industry_name']))

    stock_info = {}
    for code in main_biz_map.keys():
        ind_id = ind_name_to_id.get(ind_map[code], 0)
        prod_id = classify_main_business(main_biz_map[code], scope_map[code])
        sid = stock_to_id.get(code)
        comp_ids = competitor_map.get(sid, []) if sid is not None else []
        stock_info[code] = {
            'industry_id': ind_id,
            'product_id': prod_id,
            'competitor_ids': comp_ids,
        }

    n_new = sum(1 for c in main_biz_map if c not in stock_to_id)
    if n_new:
        print(f"[info] {n_new} 只股票未在训练集出现 → 竞对全填 -1")

    H, F = cfg.history_days, cfg.future_days
    max_comp = cfg.max_competitors
    X_list, code_list, date_list, ret_list = [], [], [], []

    for code, g in df.groupby('股票代码', sort=False):
        g = g.sort_values('交易日期')
        if len(g) < H:
            continue
        X = _apply_transform(g, FEATURE_SPECS)
        close = g['最新价'].astype(float).to_numpy(dtype=np.float64)
        dates = g['交易日期'].to_numpy()
        last_i = len(X) - H
        for i in range(last_i + 1):
            X_list.append(X[i:i + H])
            code_list.append(code)
            date_list.append(dates[i + H - 1])
            # 若存在未来 F 天数据 → 计算真实收益用于回测
            j = i + H - 1 + F
            if j < len(close) and close[i + H - 1] > 0 and close[j] > 0:
                ret_list.append(float(np.log(close[j] / close[i + H - 1])))
            else:
                ret_list.append(np.nan)

    if not X_list:
        raise RuntimeError(f"新数据无法切出样本（需至少 {H} 天连续历史）")

    X_arr = np.stack(X_list).astype(np.float32)
    X_arr = (X_arr - meta['mean']) / meta['std']

    industry_arr = np.array([stock_info[c]['industry_id'] for c in code_list], dtype=np.int64)
    product_arr = np.array([stock_info[c]['product_id'] for c in code_list], dtype=np.int64)
    comp_arr = np.full((len(X_arr), max_comp), -1, dtype=np.int64)
    for n, code in enumerate(code_list):
        comps = stock_info[code]['competitor_ids']
        if comps:
            k = min(max_comp, len(comps))
            comp_arr[n, :k] = comps[:k]

    return {
        'X': torch.from_numpy(X_arr),
        'ind': torch.from_numpy(industry_arr),
        'prod': torch.from_numpy(product_arr),
        'comp': torch.from_numpy(comp_arr),
        'symbols': code_list,
        'dates': date_list,
        'true_returns': np.array(ret_list, dtype=np.float32),
    }


@torch.no_grad()
def run_predict(model, cfg, data, batch_size):
    device = cfg.device
    ds = TensorDataset(data['X'], data['ind'], data['prod'], data['comp'])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_logits, all_ret = [], []
    for feat, ind, prod, comp in loader:
        logits, ret = model(feat.to(device), ind.to(device), prod.to(device), comp.to(device))
        all_logits.append(logits.cpu())
        all_ret.append(ret.squeeze(-1).cpu())

    logits = torch.cat(all_logits)
    return {
        'pred_trend': logits.argmax(dim=1).numpy(),
        'pred_proba': torch.softmax(logits, dim=1).numpy(),
        'pred_return': torch.cat(all_ret).numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='训练保存的 best.pt')
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--sql', type=str, default=None)
    parser.add_argument('--sql_file', type=str, default=None)
    parser.add_argument('--db', type=str, default='jhd')
    parser.add_argument('--db_config', type=str, default=None)
    parser.add_argument('--out_csv', type=str, default='predictions.csv')
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    model, cfg, meta = load_checkpoint(args.ckpt, device)
    print(f"[ckpt] tokens/seq={cfg.total_tokens}  feat_dim={cfg.stock_feat_dim}  "
          f"train_industries={len(meta['industry_name_to_id'])}  train_stocks={len(meta['stock_to_id'])}")

    df = fetch_df(args)
    print(f"[data] 输入 {len(df)} 行 × {len(df.columns)} 列")

    data = build_inference_samples(df, cfg, meta)
    print(f"[data] 构造 {len(data['X'])} 条推理样本")

    res = run_predict(model, cfg, data, args.batch_size)

    trend_names = {0: '跌', 1: '平', 2: '涨'}
    out = pd.DataFrame({
        'symbol': data['symbols'],
        'target_date': [pd.Timestamp(d).date() for d in data['dates']],
        'pred_trend': res['pred_trend'],
        'pred_trend_name': [trend_names[t] for t in res['pred_trend']],
        'pred_return': res['pred_return'],
        'proba_down': res['pred_proba'][:, 0],
        'proba_flat': res['pred_proba'][:, 1],
        'proba_up': res['pred_proba'][:, 2],
    })
    # 若新数据有未来 F 天 → 附回测列
    true_ret = data['true_returns']
    valid = ~np.isnan(true_ret)
    if valid.any():
        thr = cfg.trend_threshold
        true_trend = np.where(true_ret > thr, 2, np.where(true_ret < -thr, 0, 1))
        out['true_return'] = true_ret
        out['true_trend'] = np.where(valid, true_trend, -1)
        out['correct'] = np.where(valid, (res['pred_trend'] == true_trend).astype(int), -1)
        n_valid = valid.sum()
        acc = (res['pred_trend'][valid] == true_trend[valid]).mean()
        print(f"[backtest] {n_valid}/{len(out)} 条样本有真实未来收益，"
              f"趋势准确率={acc:.4f}")
    else:
        print("[backtest] 新数据无未来 F 天数据，仅输出预测")

    out.to_csv(args.out_csv, index=False)
    print(f"[done] 已保存 → {args.out_csv}  ({len(out)} 条)")
    print(f"[dist] 预测 涨/平/跌 = "
          f"{(out.pred_trend == 2).sum()}/{(out.pred_trend == 1).sum()}/{(out.pred_trend == 0).sum()}")


if __name__ == '__main__':
    main()