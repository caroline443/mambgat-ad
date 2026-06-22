"""
MambGAT-AD 训练脚本

用法（Windows CMD / PowerShell）：
  python train.py --config config/smap.yaml
  python train.py --config config/smap.yaml --dataset msl --epochs 50

数据准备：
  1. git clone https://github.com/khundman/telemanom
  2. 将 telemanom/data/ 整个文件夹复制到本项目 datasets/ 目录
  3. 确保以下路径存在：
       datasets/data/train/*.npy
       datasets/data/test/*.npy
       datasets/labeled_anomalies.csv
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data import build_loaders
from models import MambGATAD, PredictionLoss
from utils import evaluate_anomaly, print_metrics
from utils.threshold import PerChannelThreshold


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_args(cfg: dict, args: argparse.Namespace) -> dict:
    """命令行参数覆盖 yaml 配置"""
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.device:
        cfg["train"]["device"] = args.device
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 训练主函数
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    set_seed(cfg["train"]["seed"])

    # ── 设备 ──────────────────────────────────────────────────────
    device_str = cfg["train"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，自动切换到 CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[Info] 使用设备: {device}")

    # ── 数据 ──────────────────────────────────────────────────────
    train_loader, test_loader, test_labels, channels = build_loaders(
        data_dir       = cfg["data"]["data_dir"],
        label_file     = cfg["data"]["label_file"],
        dataset        = cfg["data"]["dataset"],
        window_size    = cfg["data"]["window_size"],
        train_step     = cfg["data"].get("window_step", 5),
        test_step      = cfg["data"].get("test_step", 1),
        batch_size     = cfg["train"]["batch_size"],
        normalize_data = cfg["data"].get("normalize", True),
        num_workers    = 0,   # Windows 下必须为 0
    )
    n_channels  = len(channels)
    window_size = cfg["data"]["window_size"]

    # ── 模型 ──────────────────────────────────────────────────────
    model = MambGATAD(
        n_channels  = n_channels,
        window_size = window_size,
        d_model     = cfg["model"]["d_model"],
        n_blocks    = cfg["model"]["n_blocks"],
        n_heads     = cfg["model"]["n_heads"],
        d_state     = cfg["model"]["d_state"],
        d_conv      = cfg["model"]["d_conv"],
        expand      = cfg["model"]["expand"],
        pred_len    = cfg["model"]["pred_len"],
        dropout     = cfg["model"]["dropout"],
    ).to(device)

    print(f"[Model] MambGAT-AD | 参数量: {model.count_parameters():,}")

    # ── 优化器 ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["train"]["lr"],
        weight_decay = cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"], eta_min=1e-6
    )
    criterion = PredictionLoss(alpha=0.5)

    # ── Checkpoint 目录 ───────────────────────────────────────────
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"best_{cfg['data']['dataset']}.pt"

    # ── 训练循环 ──────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_cnt  = 0
    patience      = cfg["train"]["patience"]

    print(f"\n{'═'*60}")
    print(f"  开始训练  |  数据集={cfg['data']['dataset'].upper()}"
          f"  |  epochs={cfg['train']['epochs']}")
    print(f"{'═'*60}")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        train_losses = []
        t0 = time.time()

        for x_batch, y_batch in tqdm(
            train_loader, desc=f"Epoch {epoch:02d}/{cfg['train']['epochs']}",
            leave=False, ncols=80
        ):
            x_batch = x_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            pred, _ = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        avg_loss = np.mean(train_losses)
        elapsed  = time.time() - t0

        # ── Validation（用训练误差收敛情况判断）────────────────
        print(f"  Epoch {epoch:02d}  loss={avg_loss:.5f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"time={elapsed:.1f}s")

        # 早停（基于 train loss，实验中可改为验证集）
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            patience_cnt  = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "cfg":         cfg,
                "channels":    channels,
            }, best_path)
            print(f"  ✓ 保存最佳模型 → {best_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n  早停触发（{patience} 轮无改善）")
                break

    # ── 评估 ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  加载最佳模型进行测试集评估 ...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # 收集训练集误差（用于拟合阈值）
    train_errors = _collect_errors(model, train_loader, device)
    # 收集测试集误差
    test_errors  = _collect_errors(model, test_loader,  device)

    # 拟合阈值
    thr_cfg = cfg.get("threshold", {})
    thresholder = PerChannelThreshold(
        method       = thr_cfg.get("method", "telemanom"),
        p            = thr_cfg.get("p", 0.13),
        error_buffer = thr_cfg.get("error_buffer", 100),
        percentile   = thr_cfg.get("percentile", 99.5),
    )
    thresholder.fit(train_errors)
    per_channel_pred, global_pred = thresholder.predict(test_errors)

    # 全局标签（任意通道异常 = 全局异常）
    test_len     = len(global_pred)
    labels_flat  = test_labels[:test_len].any(axis=1).astype(int)
    global_score = test_errors.mean(axis=1)   # 平均误差作为全局分数

    metrics = evaluate_anomaly(
        y_true=labels_flat,
        y_pred=global_pred,
        y_score=global_score,
        use_pa=True,
    )
    print_metrics(metrics, prefix=f"MambGAT-AD on {cfg['data']['dataset'].upper()}")

    # 保存评估结果
    import json
    result_path = save_dir / f"results_{cfg['data']['dataset']}.json"
    with open(result_path, "w") as f:
        json.dump({k: round(float(v), 6) for k, v in metrics.items()}, f, indent=2)
    print(f"\n  结果已保存 → {result_path}")

    return metrics


def _collect_errors(
    model: MambGATAD,
    loader,
    device: torch.device,
) -> np.ndarray:
    """推理并收集所有批次的预测误差，返回 (T, N) numpy 数组"""
    model.eval()
    all_scores = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device, dtype=torch.float32)
            _, score = model(x_batch)       # (B, N)
            all_scores.append(score.cpu().numpy())
    return np.concatenate(all_scores, axis=0)   # (T, N)


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MambGAT-AD 训练脚本")
    parser.add_argument("--config",     default="config/smap.yaml", help="配置文件路径")
    parser.add_argument("--dataset",    default=None,  choices=["smap", "msl"], help="数据集")
    parser.add_argument("--epochs",     default=None,  type=int,   help="训练轮数")
    parser.add_argument("--batch_size", default=None,  type=int,   help="批大小")
    parser.add_argument("--lr",         default=None,  type=float, help="学习率")
    parser.add_argument("--device",     default=None,  choices=["cuda", "cpu"], help="设备")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = merge_args(cfg, args)
    train(cfg)
