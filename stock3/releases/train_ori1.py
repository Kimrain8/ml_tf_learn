# train.py
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from stock3.config import OUT_DIR
from stock3.models import LightweightFusion

# ----------------- 1. 损失函数（保持非负） -----------------
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(delta=0.005)
        self.bce = nn.BCEWithLogitsLoss()
        self.huber_amp = nn.HuberLoss()

    def forward(self, mu, dir_logit, amp, y_gap, y_dir, y_amp):
        loss_gap = self.huber(mu.squeeze(-1), y_gap)
        loss_dir = self.bce(dir_logit.squeeze(-1), y_dir)
        loss_amp = self.huber_amp(amp.squeeze(-1), y_amp)
        # 加权汇总：主任务最重要，分类和振幅做正则
        return loss_gap + 0.2 * loss_dir + 0.1 * loss_amp

# ----------------- 2. 数据加载与时间切分 -----------------
def prepare_data():
    feat_dir = os.path.join(OUT_DIR, "m3_features")
    X_seq = np.load(os.path.join(feat_dir, "X_seq.npy"))
    X_tab = np.load(os.path.join(feat_dir, "X_tab.npy"))
    X_cat = np.load(os.path.join(feat_dir, "X_cat.npy"))
    y_gap = np.load(os.path.join(feat_dir, "y_gap.npy"))
    y_dir = np.load(os.path.join(feat_dir, "y_dir.npy"))
    y_amp = np.load(os.path.join(feat_dir, "y_amp.npy"))
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))

    dates = sorted(meta["trade_date"].unique())
    n_dates = len(dates)
    train_end = int(n_dates * 0.6)
    val_end = int(n_dates * 0.8)
    
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    def get_split(split_dates):
        mask = meta["trade_date"].isin(split_dates).values
        return (torch.tensor(X_seq[mask], dtype=torch.float32),
                torch.tensor(X_tab[mask], dtype=torch.float32),
                torch.tensor(X_cat[mask], dtype=torch.long),
                torch.tensor(y_gap[mask], dtype=torch.float32),
                torch.tensor(y_dir[mask], dtype=torch.float32),
                torch.tensor(y_amp[mask], dtype=torch.float32))

    return get_split(train_dates), get_split(val_dates), get_split(test_dates)

# ----------------- 3. 训练与评估 -----------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x_seq, x_tab, x_cat, y_gap, y_dir, y_amp = [b.to(device) for b in batch]
        optimizer.zero_grad()
        mu, dir_logit, amp = model(x_seq, x_tab, x_cat)
        loss = criterion(mu, dir_logit, amp, y_gap, y_dir, y_amp)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_gap)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds = {"mu": [], "dir": []}
    targets = {"gap": [], "dir": []}
    
    with torch.no_grad():
        for batch in dataloader:
            x_seq, x_tab, x_cat, y_gap, y_dir, y_amp = [b.to(device) for b in batch]
            mu, dir_logit, amp = model(x_seq, x_tab, x_cat)
            loss = criterion(mu, dir_logit, amp, y_gap, y_dir, y_amp)
            total_loss += loss.item() * len(y_gap)
            
            preds["mu"].append(mu.cpu())
            preds["dir"].append(torch.sigmoid(dir_logit).cpu())
            targets["gap"].append(y_gap.cpu())
            targets["dir"].append(y_dir.cpu())

    pred_mu = torch.cat(preds["mu"]).squeeze(-1).numpy()
    pred_dir = torch.cat(preds["dir"]).squeeze(-1).numpy()
    true_gap = torch.cat(targets["gap"]).numpy()
    true_dir = torch.cat(targets["dir"]).numpy()
    
    mae_bp = np.mean(np.abs(pred_mu - true_gap)) * 1e4
    dir_acc = np.mean((pred_dir > 0.5) == (true_dir > 0.5))
    
    # 当前市场的“自然胜率”（即盲猜跌，因为最近跌的占多数）
    base_acc = np.mean(true_dir == 0) 
    
    # 高信度预测
    high_conf_mask = pred_dir > 0.6
    if high_conf_mask.sum() > 0:
        high_conf_acc = np.mean((pred_dir[high_conf_mask] > 0.5) == (true_dir[high_conf_mask] > 0.5))
        high_conf_num = high_conf_mask.sum()
    else:
        high_conf_acc, high_conf_num = 0, 0

    return total_loss / len(dataloader.dataset), mae_bp, dir_acc, base_acc, high_conf_acc, high_conf_num

def main():
    device = torch.device("cpu")
    train_data, val_data, test_data = prepare_data()
    
    batch_size = 1024
    train_loader = DataLoader(TensorDataset(*train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_data), batch_size=batch_size, shuffle=False)

    model = LightweightFusion().to(device)
    criterion = MultiTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    epochs = 30
    best_val_loss = float("inf")
    patience_counter = 0

    print("\n开始训练 (损失应为正数且越小越好)...")
    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, mae, acc, base_acc, h_acc, h_num = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:02d} | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | "
              f"MAE: {mae:.1f}bp | 全样Acc: {acc:.2%} (基准:{base_acc:.2%}) | "
              f"高信度(>0.6)Acc: {h_acc:.2%} ({h_num}样本) | 耗时: {time.time()-t0:.1f}s")
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("\n触发早停 (连续5轮验证损失未改善，已回滚到最佳模型)。")
                break

    print("\n加载最佳模型在测试集评估...")
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth")))
    t_loss, t_mae, t_acc, t_base, t_h_acc, t_h_num = evaluate(model, test_loader, criterion, device)
    print(f"测试集 | 损失: {t_loss:.4f} | MAE: {t_mae:.1f}bp | "
          f"全样Acc: {t_acc:.2%} (基准:{t_base:.2%}) | 高信度Acc: {t_h_acc:.2%} ({t_h_num}样本)")

if __name__ == "__main__":
    main()