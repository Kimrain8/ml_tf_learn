# train.py
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
from stock3.config import OUT_DIR
from stock3.models import LightweightFusion

# ★★★ 新增：固定所有随机种子 ★★★
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(delta=0.01)

    def forward(self, mu, y_gap_excess):
        return self.huber(mu.squeeze(-1), y_gap_excess)

def prepare_data():
    feat_dir = os.path.join(OUT_DIR, "m3_features")
    X_seq = np.load(os.path.join(feat_dir, "X_seq.npy"))
    X_tab = np.load(os.path.join(feat_dir, "X_tab.npy"))
    X_cat = np.load(os.path.join(feat_dir, "X_cat.npy"))
    y_gap_excess = np.load(os.path.join(feat_dir, "y_gap_excess.npy"))
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))

    valid_mask = meta["label_valid"] == 1
    meta_train = meta[valid_mask].copy()
    X_seq_train = X_seq[valid_mask]
    X_tab_train = X_tab[valid_mask]
    X_cat_train = X_cat[valid_mask]
    y_train = y_gap_excess[valid_mask]

    dates = sorted(meta_train["trade_date"].unique())
    n = len(dates)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    def get_split(ds):
        m = meta_train["trade_date"].isin(ds).values
        return [torch.tensor(a[m], dtype=t) for a, t in [
            (X_seq_train, torch.float32), (X_tab_train, torch.float32), (X_cat_train, torch.long),
            (y_train, torch.float32)]]

    return (get_split(dates[:train_end]),
            get_split(dates[train_end:val_end]),
            get_split(dates[val_end:]),
            meta_train, dates)

def train_epoch(model, dl, crit, opt, dev):
    model.train()
    tot = 0
    for x_seq, x_tab, x_cat, y_gap_excess in dl:
        x_seq, x_tab, x_cat, y_gap_excess = x_seq.to(dev), x_tab.to(dev), x_cat.to(dev), y_gap_excess.to(dev)
        opt.zero_grad()
        mu, _ = model(x_seq, x_tab, x_cat)
        loss = crit(mu, y_gap_excess)
        loss.backward()
        opt.step()
        tot += loss.item() * len(y_gap_excess)
    return tot / len(dl.dataset)

def evaluate(model, dl, crit, dev, eval_meta=None):
    model.eval()
    tot = 0
    preds = []
    with torch.no_grad():
        for x_seq, x_tab, x_cat, y_gap_excess in dl:
            x_seq, x_tab, x_cat = x_seq.to(dev), x_tab.to(dev), x_cat.to(dev)
            mu, _ = model(x_seq, x_tab, x_cat)
            loss = crit(mu, y_gap_excess.to(dev))
            tot += loss.item() * len(y_gap_excess)
            preds.append(mu.cpu().numpy().ravel())

    pred_all = np.concatenate(preds)
    loss_avg = tot / len(dl.dataset)

    ic = 0
    if eval_meta is not None:
        eval_meta = eval_meta.copy()
        eval_meta["pred"] = pred_all
        ic_list = []
        for d in sorted(eval_meta["trade_date"].unique()):
            day_data = eval_meta[eval_meta["trade_date"] == d]
            if len(day_data) > 10:
                ic_val, _ = spearmanr(day_data["pred"], day_data["y_gap_excess"])
                if not np.isnan(ic_val):
                    ic_list.append(ic_val)
        if ic_list:
            ic = np.mean(ic_list)

    return loss_avg, ic

def main():
    set_seed(42)  # 固定随机种子
    
    dev = torch.device("cpu")
    tr, va, te, meta_train, dates = prepare_data()
    batch_size = 2048
    tr_dl = DataLoader(TensorDataset(*tr), batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(TensorDataset(*va), batch_size=batch_size, shuffle=False)
    te_dl = DataLoader(TensorDataset(*te), batch_size=batch_size, shuffle=False)

    model = LightweightFusion(seq_dim=11, tab_dim=27, cat_vocab_size=400).to(dev)
    crit = MultiTaskLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)

    print("\n训练 15 轮（按验证集 Rank IC 保存最佳模型）...")
    best_val_ic = -1.0
    model_path = os.path.join(OUT_DIR, "best_model.pth")

    va_dates = dates[int(len(dates)*0.6):int(len(dates)*0.8)]
    va_meta = meta_train[meta_train["trade_date"].isin(va_dates)]

    for ep in range(15):
        t0 = time.time()
        tl = train_epoch(model, tr_dl, crit, opt, dev)
        vl, _ = evaluate(model, va_dl, crit, dev)
        _, va_ic = evaluate(model, va_dl, crit, dev, va_meta)

        print(f"Ep {ep+1:02d} | train {tl:.5f} | val {vl:.5f} | Val Rank IC {va_ic:.4f} | {time.time()-t0:.1f}s")

        # ★★★ 只有验证集 Rank IC 提升才保存模型 ★★★
        if va_ic > best_val_ic:
            best_val_ic = va_ic
            torch.save(model.state_dict(), model_path)
            print(f"   -> 发现更好的 IC ({best_val_ic:.4f})，已保存模型")

    print("\n测试集评估：")
    model.load_state_dict(torch.load(model_path, map_location=dev))
    te_dates = dates[int(len(dates)*0.8):]
    te_meta = meta_train[meta_train["trade_date"].isin(te_dates)]
    _, te_ic = evaluate(model, te_dl, crit, dev, te_meta)
    print(f"测试集 Rank IC: {te_ic:.4f}")
    print(f"[保存] 最佳模型: {model_path}")

if __name__ == "__main__":
    main()