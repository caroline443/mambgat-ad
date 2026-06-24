"""
Telemanom Baseline — 使用我们的数据和评估框架复现

原论文：Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs
        and Nonparametric Dynamic Thresholding", KDD 2018

核心思路：
  - 对每个通道训练一个 LSTM 预测模型
  - 用预测残差作为异常分数
  - Telemanom 动态阈值

用法：
  python baselines/telemanom_baseline.py --config config/smap.yaml
"""

from __future__ import annotations

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import load_at_format, load_telemanom_format, get_channels
from utils.metrics import evaluate_per_channel, evaluate_anomaly, print_metrics
from utils.threshold import PercentileThreshold


# ─────────────────────────────────────────────────────────────────────────────
# 单通道 LSTM 预测模型
# ─────────────────────────────────────────────────────────────────────────────

class ChannelLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden: int = 64,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B, T, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)  # (B,)


def make_windows(series: np.ndarray, window: int = 100, step: int = 1
                 ) -> tuple:
    X, y = [], []
    for i in range(0, len(series) - window, step):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_channel(train_data: np.ndarray, window: int = 100,
                  epochs: int = 20, device: str = "cuda") -> tuple:
    """训练单通道 LSTM，返回训练集残差"""
    X_tr, y_tr = make_windows(train_data, window, step=1)
    ds = TensorDataset(torch.from_numpy(X_tr).unsqueeze(-1),
                       torch.from_numpy(y_tr))
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    model = ChannelLSTM().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

    # 训练集误差
    model.eval()
    errors = []
    with torch.no_grad():
        for xb, yb in DataLoader(ds, batch_size=512):
            pred = model(xb.to(device)).cpu().numpy()
            errors.append(np.abs(pred - yb.numpy()))
    return model, np.concatenate(errors)


def eval_channel(model, test_data: np.ndarray, window: int = 100,
                 device: str = "cuda") -> np.ndarray:
    """返回测试集残差"""
    X_te, y_te = make_windows(test_data, window, step=1)
    ds = TensorDataset(torch.from_numpy(X_te).unsqueeze(-1),
                       torch.from_numpy(y_te))
    model.eval()
    errors = []
    with torch.no_grad():
        for xb, yb in DataLoader(ds, batch_size=512):
            pred = model(xb.to(device)).cpu().numpy()
            errors.append(np.abs(pred - yb.numpy()))
    return np.concatenate(errors)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fmt    = cfg["data"].get("format", "AT").upper()
    window = cfg["data"].get("window_size", 100)
    pct    = cfg.get("threshold", {}).get("percentile", 99.5)
    dname  = cfg["data"]["dataset"].upper()

    print(f"\n{'═'*60}")
    print(f"  Telemanom Baseline | {dname} | 格式={fmt}")
    print(f"{'═'*60}\n")

    # 加载数据
    if fmt == "AT":
        train, test, global_label = load_at_format(
            cfg["data"]["data_dir"], cfg["data"]["dataset"])
        n_ch = train.shape[1]
        per_ch_labels = None   # AT 格式只有全局标签
    else:
        train, test, per_ch_labels, channels = load_telemanom_format(
            cfg["data"]["data_dir"], cfg["data"]["label_file"],
            cfg["data"]["dataset"])
        n_ch = train.shape[1]

    test_len = len(test) - window

    # 逐通道训练
    all_train_err = np.zeros((len(train) - window, n_ch))
    all_test_err  = np.zeros((test_len, n_ch))

    for i in range(n_ch):
        print(f"  通道 {i+1}/{n_ch} ...", end="\r")
        model, tr_err = train_channel(train[:, i], window, epochs=15, device=device)
        te_err        = eval_channel(model, test[:, i], window, device)
        l_tr = min(len(tr_err), len(all_train_err))
        l_te = min(len(te_err), test_len)
        all_train_err[:l_tr, i] = tr_err[:l_tr]
        all_test_err[:l_te, i]  = te_err[:l_te]

    print()

    # 评估
    if fmt == "AT":
        global_score = all_test_err.mean(1)
        thr  = float(np.percentile(all_train_err.mean(1), pct))
        pred = (global_score > thr).astype(int)
        m = evaluate_anomaly(global_label[:test_len].astype(int),
                             pred, global_score, use_pa=True,
                             dataset=cfg['data']['dataset'])
        print_metrics(m, f"Telemanom Baseline | {dname} [全局]")
    else:
        m = evaluate_per_channel(
            per_ch_labels[:test_len], all_test_err, all_train_err, pct)
        print_metrics(m, f"Telemanom Baseline | {dname} [逐通道宏平均]")

    # 保存结果
    import json, os
    os.makedirs("checkpoints", exist_ok=True)
    out = f"checkpoints/baseline_telemanom_{cfg['data']['dataset']}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: round(float(v), 6) for k, v in m.items()
                   if isinstance(v, (int, float))}, f, indent=2)
    print(f"  结果已保存 → {out}")

    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/smap.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
