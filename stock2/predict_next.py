"""
预测「每只股票最后 H 天 → 下一个交易日」的结果，并导出注意力权重。

输出:
  predictions/predictions_next.csv
      symbol, d1_date, target_date, pred_trend, pred_trend_name,
      pred_return, proba_down, proba_flat, proba_up
  predictions/attention_next.csv
      symbol, d1_date, CLS, IND, PROD, COMP1..5, PATCH1..N, SEP
      （每列是该 token 在 CLS 聚合时的平均注意力权重，越大越重要）
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from pandas.tseries.offsets import BDay

from config import Config
from model import StockTransformer
from data_pipeline import (
    FEATURE_SPECS, SW_COLS, DB_COLUMN_MAP, DEFAULT_STOCK_SQL,
    _apply_transform, _pick_industry_name, _to_str,
)
from business_classifier import classify_main_business


def _token_names(cfg):
    """与 tokenizers.py 的序列顺序一致"""
    names = ['CLS', 'IND', 'PROD']
    names.extend([f'COMP{i+1}' for i in range(cfg.max_competitors)])
    names.extend([f'PATCH{i+1}' for i in range(cfg.n_patches)])
    names.append('SEP')
    return names


def load_checkpoint(path, device):
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
            print("[data] 使用内置 DEFAULT_STOCK_SQL")
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


def build_last_samples(df, cfg, meta):
    """每只股票取最后 H 天作为输入窗口"""
    stock_to_id = meta['stock_to_id']
    ind_name_to_id = meta['industry_name_to_id']
    competitor_map = meta.get('competitor_map', {})

    for key, col in SW_COLS.items():
        if col not in df.columns:
            df[col] = '未知'

    H = cfg.history_days
    X_list, code_list, d1_list, ind_list, prod_list, comp_list = [], [], [], [], [], []
    n_short = 0
    n_unknown_ind = 0

    for code, g in df.groupby('股票代码', sort=False):
        g = g.sort_values('交易日期')
        if len(g) < H:
            n_short += 1
            continue

        sub = g.iloc[-H:]
        X = _apply_transform(sub, FEATURE_SPECS)   # (H, D)
        X_list.append(X)

        d1 = pd.Timestamp(sub['交易日期'].iloc[-1])

        sw3 = _to_str(sub[SW_COLS['sw3']].iloc[-1])
        sw2 = _to_str(sub[SW_COLS['sw2']].iloc[-1])
        sw1 = _to_str(sub[SW_COLS['sw1']].iloc[-1])
        ind_name = _pick_industry_name(sw3, sw2, sw1)
        if ind_name in ind_name_to_id:
            ind_id = ind_name_to_id[ind_name]
        else:
            ind_id = 0
            n_unknown_ind += 1

        main_biz = _to_str(sub['主营业务'].iloc[-1]) if '主营业务' in sub.columns else ''
        scope = _to_str(sub['经营范围'].iloc[-1]) if '经营范围' in sub.columns else ''
        prod_id = classify_main_business(main_biz, scope)

        sid = stock_to_id.get(code)
        comps = competitor_map.get(sid, []) if sid is not None else []

        code_list.append(code)
        d1_list.append(d1)
        ind_list.append(ind_id)
        prod_list.append(prod_id)
        comp_list.append(comps)

    if not X_list:
        raise RuntimeError(f"无可用样本：所有股票历史不足 {H} 天")

    X_arr = np.stack(X_list).astype(np.float32)
    X_arr = (X_arr - meta['mean']) / meta['std']

    ind_arr = np.array(ind_list, dtype=np.int64)
    prod_arr = np.array(prod_list, dtype=np.int64)
    comp_arr = np.full((len(X_arr), cfg.max_competitors), -1, dtype=np.int64)
    for n, comps in enumerate(comp_list):
        if comps:
            k = min(cfg.max_competitors, len(comps))
            comp_arr[n, :k] = comps[:k]

    print(f"[data] 构造 {len(X_arr)} 条预测样本（不足 {H} 天跳过 {n_short} 只）")
    if n_unknown_ind:
        print(f"[warn] {n_unknown_ind} 只股票行业未在训练集出现 → id=0")

    return {
        'X': torch.from_numpy(X_arr),
        'ind': torch.from_numpy(ind_arr),
        'prod': torch.from_numpy(prod_arr),
        'comp': torch.from_numpy(comp_arr),
        'symbols': code_list,
        'd1_dates': d1_list,
    }


@torch.no_grad()
def run_predict(model, cfg, data, batch_size):
    device = cfg.device
    N = data['X'].size(0)
    logits_chunks, ret_chunks, attn_chunks = [], [], []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        logits, ret, attn = model(
            data['X'][start:end].to(device),
            data['ind'][start:end].to(device),
            data['prod'][start:end].to(device),
            data['comp'][start:end].to(device),
            return_attn=True,
        )
        logits_chunks.append(logits.cpu())
        ret_chunks.append(ret.squeeze(-1).cpu())
        attn_chunks.append(attn.cpu())

    logits = torch.cat(logits_chunks)
    return {
        'pred_trend': logits.argmax(dim=1).numpy(),
        'pred_proba': torch.softmax(logits, dim=1).numpy(),
        'pred_return': torch.cat(ret_chunks).numpy(),
        'attn': torch.cat(attn_chunks).numpy(),   # (N, T)
    }


def print_insights(pred_df, attn_df, token_names):
    """核心结论 - 每次跑完自动打印"""
    print("\n" + "=" * 64)
    print("【核心结论】")
    print("=" * 64)

    n_up = int((pred_df.pred_trend == 2).sum())
    n_flat = int((pred_df.pred_trend == 1).sum())
    n_down = int((pred_df.pred_trend == 0).sum())
    total = len(pred_df)
    print(f"1) 全市场预测分布: 涨 {n_up} / 平 {n_flat} / 跌 {n_down}  (共 {total})")
    print(f"   上涨占比 {n_up/total:.1%}  下跌占比 {n_down/total:.1%}")

    avg_ret = float(pred_df.pred_return.mean())
    med_ret = float(pred_df.pred_return.median())
    print(f"2) 预测收益: 均值 {avg_ret:+.4%}  中位数 {med_ret:+.4%}  （单日对数）")

    print(f"3) 看涨 Top10（代码 / 预测收益 / 上涨概率）:")
    for _, r in pred_df.nlargest(10, 'pred_return').iterrows():
        print(f"     {r['symbol']:<12s}  {r['pred_return']:+.4f}  P(up)={r['proba_up']:.3f}")

    print(f"4) 看跌 Top10（代码 / 预测收益 / 下跌概率）:")
    for _, r in pred_df.nsmallest(10, 'pred_return').iterrows():
        print(f"     {r['symbol']:<12s}  {r['pred_return']:+.4f}  P(down)={r['proba_down']:.3f}")

    print(f"5) 全市场平均注意力（越大 = 模型越依赖该类信息）:")
    avg_attn = attn_df[token_names].mean(0)

    # 按类型分组汇总
    groups = {'行业 IND': ['IND'], '产品 PROD': ['PROD'],
              '竞争对手 COMP': [n for n in token_names if n.startswith('COMP')],
              '历史走势 PATCH': [n for n in token_names if n.startswith('PATCH')]}
    group_sums = {name: float(avg_attn[cols].sum()) for name, cols in groups.items()}
    total_attn = sum(group_sums.values()) + 1e-12
    for name, v in sorted(group_sums.items(), key=lambda x: -x[1]):
        pct = v / total_attn
        bar = '█' * max(1, int(pct * 40))
        print(f"     {name:<14s} {v:.4f}  ({pct:5.1%})  {bar}")

    print(f"6) 单个 token 平均注意力 Top5:")
    single_attn = avg_attn.drop(['CLS', 'SEP']).sort_values(ascending=False)
    for name, v in single_attn.head(5).items():
        print(f"     {name:<10s}  {v:.4f}")

    print("\n【结论解读】")
    top_group = max(group_sums, key=group_sums.get)
    print(f"  → 当前模型预测 '{top_group}' 依赖度最高")
    if top_group == '历史走势 PATCH':
        print(f"    说明模型主要从历史价格模式预测，行业/竞对信号较弱")
    elif top_group == '竞争对手 COMP':
        print(f"    说明模型的预测高度依赖同行业竞争股票联动")
    else:
        print(f"    说明模型主要依赖股票自身行业或业务属性")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--sql', type=str, default=None)
    parser.add_argument('--sql_file', type=str, default=None)
    parser.add_argument('--db', type=str, default='jhd')
    parser.add_argument('--db_config', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='predictions')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--target_date', type=str, default=None,
                        help='目标日期（不填=最后一天+1工作日）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    model, cfg, meta = load_checkpoint(args.ckpt, device)
    print(f"[ckpt] future_days={cfg.future_days}  history_days={cfg.history_days}  "
          f"tokens/seq={cfg.total_tokens}  train_stocks={len(meta['stock_to_id'])}")

    if cfg.future_days != 1:
        print(f"⚠️ [warn] future_days={cfg.future_days}，本次预测是累计收益，不是'下一天'")

    df = fetch_df(args)
    print(f"[data] 输入 {len(df)} 行 × {len(df.columns)} 列")

    data = build_last_samples(df, cfg, meta)
    res = run_predict(model, cfg, data, args.batch_size)

    trend_names = {0: '跌', 1: '平', 2: '涨'}

    # 目标日：取所有 d1_date 的最大值 + 1 个工作日
    max_d1 = max(data['d1_dates'])
    auto_target = (pd.Timestamp(max_d1) + BDay(1)).date()
    target_date = args.target_date if args.target_date else str(auto_target)

    pred_df = pd.DataFrame({
        'symbol': data['symbols'],
        'd1_date': [pd.Timestamp(d).date() for d in data['d1_dates']],
        'target_date': target_date,
        'pred_trend': res['pred_trend'],
        'pred_trend_name': [trend_names[t] for t in res['pred_trend']],
        'pred_return': res['pred_return'],
        'proba_down': res['pred_proba'][:, 0],
        'proba_flat': res['pred_proba'][:, 1],
        'proba_up': res['pred_proba'][:, 2],
    })

    os.makedirs(args.out_dir, exist_ok=True)
    pred_path = os.path.join(args.out_dir, 'predictions_next.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"[done] 预测结果 → {pred_path}  ({len(pred_df)} 条)")

    token_names = _token_names(cfg)
    attn_df = pd.DataFrame(res['attn'], columns=token_names)
    attn_df.insert(0, 'd1_date', [pd.Timestamp(d).date() for d in data['d1_dates']])
    attn_df.insert(0, 'symbol', data['symbols'])
    attn_path = os.path.join(args.out_dir, 'attention_next.csv')
    attn_df.to_csv(attn_path, index=False)
    print(f"[done] 注意力结果 → {attn_path}  ({len(attn_df)} 条 × {len(token_names)} token)")

    print_insights(pred_df, attn_df, token_names)


if __name__ == '__main__':
    main()