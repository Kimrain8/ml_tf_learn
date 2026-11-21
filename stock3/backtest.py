# backtest.py
import os
import numpy as np
import pandas as pd
import torch
from config import OUT_DIR
from models import LightweightFusion

def run_backtest(select_ratio=0.04):
    """
    select_ratio: 每天选出的股票比例（例如 0.05 就是选排名前 5% 的股票）
    """
    dev = torch.device("cpu")
    feat_dir = os.path.join(OUT_DIR, "m3_features")
    
    # 1. 加载数据
    X_seq = np.load(os.path.join(feat_dir, "X_seq.npy"))
    X_tab = np.load(os.path.join(feat_dir, "X_tab.npy"))
    X_cat = np.load(os.path.join(feat_dir, "X_cat.npy"))
    y_gap = np.load(os.path.join(feat_dir, "y_gap.npy"))
    y_gap_excess = np.load(os.path.join(feat_dir, "y_gap_excess.npy"))
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))

    # 2. 严格划分测试集（只取最后 20% 且 label_valid == 1 的数据）
    valid_mask = meta["label_valid"] == 1
    meta_valid = meta[valid_mask].copy()
    dates = sorted(meta_valid["trade_date"].unique())
    n = len(dates)
    
    # 取最后 20% 作为测试集
    test_dates = dates[int(n * 0.8):]
    test_mask = meta_valid["trade_date"].isin(test_dates).values
    
    test_meta = meta_valid[test_mask].copy()
    X_seq_test = X_seq[valid_mask][test_mask]
    X_tab_test = X_tab[valid_mask][test_mask]
    X_cat_test = X_cat[valid_mask][test_mask]
    y_gap_test = y_gap[valid_mask][test_mask]
    y_excess_test = y_gap_excess[valid_mask][test_mask]

    # 3. 加载模型并预测
    model = LightweightFusion(seq_dim=11, tab_dim=27, cat_vocab_size=400).to(dev)
    model_path = os.path.join(OUT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.eval()

    with torch.no_grad():
        x_seq_t = torch.tensor(X_seq_test, dtype=torch.float32)
        x_tab_t = torch.tensor(X_tab_test, dtype=torch.float32)
        x_cat_t = torch.tensor(X_cat_test, dtype=torch.long)
        pred_excess, _ = model(x_seq_t, x_tab_t, x_cat_t)
        pred_excess = pred_excess.squeeze(-1).numpy()

    test_meta["pred"] = pred_excess
    test_meta["true_y_gap"] = y_gap_test
    test_meta["true_excess"] = y_excess_test

    print(f"===== 回测开始 (阈值: 前 {select_ratio*100:.1f}%) =====")
    print(f"测试集范围: {test_dates[0].date()} ~ {test_dates[-1].date()} (共 {len(test_dates)} 天)\n")

    daily_win_days = 0
    total_top_stocks = 0
    total_top_wins = 0
    total_excess_bp = 0

    # 4. 按天回测
    for d in test_dates:
        day_data = test_meta[test_meta["trade_date"] == d].copy()
        if len(day_data) < 10: continue

        # 排序选出 Top N%
        day_data = day_data.sort_values("pred", ascending=False)
        n_select = max(1, int(len(day_data) * select_ratio))
        top_stocks = day_data.head(n_select)

        # 当日大盘基准（绝对收益均值）
        market_mean_gap = day_data["true_y_gap"].mean()
        
        # 我们选出的股票的绝对收益均值
        top_mean_gap = top_stocks["true_y_gap"].mean()
        
        # 实际跑赢大盘的幅度
        excess_bp = (top_mean_gap - market_mean_gap) * 1e4
        total_excess_bp += excess_bp

        # 单票胜率：选出股票中，真实超额收益 > 0 的比例
        win_stocks = (top_stocks["true_excess"] > 0).sum()
        win_rate = win_stocks / len(top_stocks)

        total_top_stocks += len(top_stocks)
        total_top_wins += win_stocks

        # 日胜率：选出的股票平均表现是否跑赢了全市场
        if top_mean_gap > market_mean_gap:
            daily_win_days += 1
            day_result = "✅ 跑赢大盘"
        else:
            day_result = "❌ 跑输大盘"

        print(f"日期: {d.date()} | 选股数: {n_select:3d}只 | "
              f"单票胜率: {win_rate:.1%} | 超额收益: {excess_bp:+.1f}bp | {day_result}")

    # 5. 输出总览
    print("\n" + "="*50)
    print("回测总结报告：")
    print("="*50)
    print(f"策略日胜率 (每天平均是否跑赢): {(daily_win_days / len(test_dates)):.1%} ({daily_win_days}/{len(test_dates)} 天)")
    print(f"单票胜率   (选出股票跑赢比例): {(total_top_wins / total_top_stocks):.1%} ({total_top_wins}/{total_top_stocks} 只)")
    print(f"平均每日超额收益 (Alpha):      {total_excess_bp / len(test_dates):+.1f} bp")
    print("="*50)
    print("* 注：正数代表选出的股票平均开盘强于大盘。+10bp 表示平均每天开盘多赚 0.1%。")

if __name__ == "__main__":
    # 你可以修改这里来测试不同的选股比例 (0.05 代表 5%)
    run_backtest(select_ratio=0.01)