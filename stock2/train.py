import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from model import StockTransformer
from dataset import StockDataset
from data_pipeline import build_stock_dataset, DEFAULT_STOCK_SQL


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights_from_returns(ret, thr, n_classes=3, device='cpu'):
    trend = np.ones(len(ret), dtype=np.int64)
    trend[ret > thr] = 2
    trend[ret < -thr] = 0
    counts = np.bincount(trend, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = len(trend) / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def run_epoch(model, loader, optimizer, ce_loss, mse_loss, lambda_reg, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    mse_sum = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for feat, ind, prod, comp, trend, ret in loader:
            feat, ind, prod, comp, trend, ret = [
                x.to(device) for x in (feat, ind, prod, comp, trend, ret)
            ]
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits, pred_ret = model(feat, ind, prod, comp)
            loss = ce_loss(logits, trend) + lambda_reg * mse_loss(pred_ret.squeeze(-1), ret)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n = len(trend)
            total_loss += loss.item() * n
            pred = logits.argmax(dim=1)
            correct += (pred == trend).sum().item()
            total += n
            mse_sum += ((pred_ret.squeeze(-1) - ret) ** 2).sum().item()

    return total_loss / total, correct / total, mse_sum / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--sql', type=str, default=None)
    parser.add_argument('--sql_file', type=str, default=None)
    parser.add_argument('--db', type=str, default='jhd')
    parser.add_argument('--db_config', type=str, default=None)
    parser.add_argument('--out', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--main_biz_col', type=str, default='主营业务')
    parser.add_argument('--biz_scope_col', type=str, default='经营范围')
    parser.add_argument('--val_days', type=int, default=0,
                        help='最后 N 个交易日作为验证集；0=无验证集（默认）')
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    set_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    device = cfg.device
    print(f"[device] {device}")

    if args.csv:
        print(f"[data] 模式=CSV  路径={args.csv}")
        train_data, val_data, meta = build_stock_dataset(
            cfg, csv_path=args.csv, val_days=args.val_days, seed=cfg.seed,
            main_biz_col=args.main_biz_col, biz_scope_col=args.biz_scope_col,
        )
    else:
        sql = args.sql
        if sql is None and args.sql_file:
            with open(args.sql_file, 'r', encoding='utf-8') as f:
                sql = f.read()
        if sql is None:
            sql = DEFAULT_STOCK_SQL
            print("[data] 模式=DB（使用内置 DEFAULT_STOCK_SQL）")
        else:
            print("[data] 模式=DB（自定义 SQL）")
        train_data, val_data, meta = build_stock_dataset(
            cfg, sql=sql, db=args.db, db_config_path=args.db_config,
            val_days=args.val_days, seed=cfg.seed,
            main_biz_col=args.main_biz_col, biz_scope_col=args.biz_scope_col,
        )

    has_val = val_data is not None and meta['n_val'] > 0
    print(f"[data] train={meta['n_train']}  val={meta['n_val'] if has_val else 0}  "
          f"feat_dim={cfg.stock_feat_dim}")
    print(f"[data] industries={cfg.num_industries}  products={cfg.num_products}  stocks={cfg.total_stocks}")

    import pandas as pd
    pd.DataFrame([
        {'industry_id': i, 'name': meta['industry_id_to_name'][i]}
        for i in range(cfg.num_industries)
    ]).to_csv(os.path.join(args.out, 'industry_id_map.csv'), index=False)

    train_ds = StockDataset(cfg, train_data)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = None
    if has_val:
        val_ds = StockDataset(cfg, val_data)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = StockTransformer(cfg).to(device)
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}  tokens/seq={cfg.total_tokens}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max' if has_val else 'min',
        factor=0.5, patience=3, min_lr=1e-6,
    )

    cw = class_weights_from_returns(train_data['future_return'], cfg.trend_threshold, device=device)
    print(f"[loss] class_weights={cw.tolist()}")
    ce_loss = nn.CrossEntropyLoss(weight=cw)
    mse_loss = nn.MSELoss()

    best_score = -1.0 if has_val else float('inf')
    bad_epochs = 0
    best_path = os.path.join(args.out, 'best.pt')

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc, train_mse = run_epoch(
            model, train_loader, optimizer, ce_loss, mse_loss, cfg.lambda_reg, device, train=True,
        )

        if has_val:
            val_loss, val_acc, val_mse = run_epoch(
                model, val_loader, optimizer, ce_loss, mse_loss, cfg.lambda_reg, device, train=False,
            )
            scheduler.step(val_acc)
            print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | val_mse {val_mse:.6f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")
            improved = val_acc > best_score
            score = val_acc
        else:
            scheduler.step(train_loss)
            print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                  f"train_mse {train_mse:.6f} | lr {optimizer.param_groups[0]['lr']:.2e}")
            improved = train_loss < best_score
            score = train_loss

        if improved:
            best_score = score
            bad_epochs = 0
            torch.save(
                {'model': model.state_dict(), 'meta': meta,
                 'cfg': {k: v for k, v in cfg.__dict__.items() if k != 'device'}},
                best_path,
            )
            print(f"  ↳ best saved (score={best_score:.4f}) -> {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"[early stop] no improvement for {cfg.patience} epochs")
                break

    print(f"[done] best_score={best_score:.4f}")


if __name__ == '__main__':
    main()