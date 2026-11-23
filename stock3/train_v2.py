# train_v2.py
import os, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
from config import OUT_DIR
from models_v2 import LightweightFusionV2

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(delta=0.01)
    def forward(self, mu, y_gap_excess):
        return self.huber(mu.squeeze(-1), y_gap_excess)

def prepare_data():
    feat_dir = os.path.join(OUT_DIR, "m3_features_v2")
    X_seq = np.load(os.path.join(feat_dir, "X_seq.npy"))
    X_ind_seq = np.load(os.path.join(feat_dir, "X_ind_seq.npy"))
    X_tab = np.load(os.path.join(feat_dir, "X_tab.npy"))
    X_cat = np.load(os.path.join(feat_dir, "X_cat.npy"))
    y_gap_excess = np.load(os.path.join(feat_dir, "y_gap_excess.npy"))
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))

    valid_mask = meta["label_valid"] == 1
    meta_train = meta[valid_mask].copy()
    
    dates = sorted(meta_train["trade_date"].unique())
    n = len(dates)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    def get_split(ds):
        m = meta_train["trade_date"].isin(ds).values
        return [torch.tensor(a[valid_mask][m], dtype=t) for a, t in [
            (X_seq, torch.float32), (X_ind_seq, torch.float32), 
            (X_tab, torch.float32), (X_cat, torch.long), (y_gap_excess, torch.float32)]]
    
    latest_date = str(meta["trade_date"].max()).split()[0].replace("-", "")
    return get_split(dates[:train_end]), get_split(dates[train_end:val_end]), get_split(dates[val_end:]), meta_train, dates, latest_date

def train_epoch(model, dl, crit, opt, dev):
    model.train(); tot = 0
    for x_seq, x_ind_seq, x_tab, x_cat, y in dl:
        x_seq, x_ind_seq, x_tab, x_cat, y = x_seq.to(dev), x_ind_seq.to(dev), x_tab.to(dev), x_cat.to(dev), y.to(dev)
        opt.zero_grad()
        mu, _ = model(x_seq, x_ind_seq, x_tab, x_cat)
        loss = crit(mu, y)
        loss.backward(); opt.step()
        tot += loss.item() * len(y)
    return tot / len(dl.dataset)

def evaluate(model, dl, crit, dev, eval_meta=None):
    model.eval(); tot = 0; preds = []
    with torch.no_grad():
        for x_seq, x_ind_seq, x_tab, x_cat, y in dl:
            x_seq, x_ind_seq, x_tab, x_cat = x_seq.to(dev), x_ind_seq.to(dev), x_tab.to(dev), x_cat.to(dev)
            mu, _ = model(x_seq, x_ind_seq, x_tab, x_cat)
            loss = crit(mu, y.to(dev))
            tot += loss.item() * len(y)
            preds.append(mu.cpu().numpy().ravel())
            
    pred_all = np.concatenate(preds)
    ic = 0
    if eval_meta is not None:
        eval_meta = eval_meta.copy()
        eval_meta["pred"] = pred_all
        ic_list = []
        for d in sorted(eval_meta["trade_date"].unique()):
            day_data = eval_meta[eval_meta["trade_date"] == d]
            if len(day_data) > 10:
                ic_val, _ = spearmanr(day_data["pred"], day_data["y_gap_excess"])
                if not np.isnan(ic_val): ic_list.append(ic_val)
        if ic_list: ic = np.mean(ic_list)
    return tot / len(dl.dataset), ic

def main():
    set_seed(42)
    dev = torch.device("cpu")
    tr, va, te, meta_train, dates, latest_date = prepare_data()
    
    tr_dl = DataLoader(TensorDataset(*tr), batch_size=2048, shuffle=True)
    va_dl = DataLoader(TensorDataset(*va), batch_size=2048, shuffle=False)
    te_dl = DataLoader(TensorDataset(*te), batch_size=2048, shuffle=False)

    model = LightweightFusionV2(seq_dim=11, ind_seq_dim=3, tab_dim=28, cat_vocab_size=400).to(dev)
    crit = MultiTaskLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)

    print(f"\n开始训练 LSTM (按日保存，本次运行标记为 {latest_date})...")
    best_val_ic = -1.0
    
    # ★ 模型保存在 m3_features_v2 目录下，带上日期
    model_path = os.path.join(OUT_DIR, "m3_features_v2", f"lstm_{latest_date}.pth")
    
    va_meta = meta_train[meta_train["trade_date"].isin(dates[int(len(dates)*0.6):int(len(dates)*0.8)])]

    for ep in range(15):
        t0 = time.time()
        tl = train_epoch(model, tr_dl, crit, opt, dev)
        vl, _ = evaluate(model, va_dl, crit, dev)
        _, va_ic = evaluate(model, va_dl, crit, dev, va_meta)
        
        print(f"Ep {ep+1:02d} | train {tl:.5f} | val {vl:.5f} | Val Rank IC {va_ic:.4f} | {time.time()-t0:.1f}s")
        
        if va_ic > best_val_ic:
            best_val_ic = va_ic
            torch.save(model.state_dict(), model_path)
            print(f"   -> 发现更好的 IC ({best_val_ic:.4f})，已保存模型至: {model_path}")

    print("\n测试集评估：")
    model.load_state_dict(torch.load(model_path, map_location=dev))
    te_meta = meta_train[meta_train["trade_date"].isin(dates[int(len(dates)*0.8):])]
    _, te_ic = evaluate(model, te_dl, crit, dev, te_meta)
    print(f"测试集 Rank IC: {te_ic:.4f}")

if __name__ == "__main__":
    main()