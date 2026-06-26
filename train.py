"""
MambGAT-AD 训练脚本（重构版）

用法：
  python train.py --config config/smap.yaml
  python train.py --config config/msl.yaml

渐进式开发说明：
  每次只改 models/__init__.py 里的导入版本（v0 → v1 → v2 ...）
  训练脚本本身不需要改动。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# 数据和工具（复用现有模块）
from data.dataset import TimeSeriesDataset, build_loaders
from models import MambGATAD, AnomalyLoss
import models as _models_pkg
from utils.metrics import (
    evaluate_anomaly, print_metrics,
    roc_auc_score, point_adjust,
    anomaly_ratio_threshold,
)


# ─────────────────────────────────────────────────────────────────────────────
# 工具
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


# ─────────────────────────────────────────────────────────────────────────────
# 推理：收集异常分数
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_scores(model: MambGATAD, loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    对 loader 中所有窗口推理，返回 (T, N) 异常分数矩阵。

    score[i, n] = |pred[i,n] - y[i,n]| + |recon[i,-1,n] - x[i,-1,n]|

    注：recon 只取最后一步（ContrastAD 代码 test() 函数同协议）：
        point_loss = criterion(input_seq[:, -1, :], recons[:, -1, :])
        使用全窗口均值会稀释信号。
    """
    model.eval()
    all_scores = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.to(device, dtype=torch.float32)

        pred, recon, _, _ = model(x_batch)

        pred_err  = (pred.squeeze(-1) - y_batch).abs()              # (B, N)
        recon_err = (recon[:, -1, :] - x_batch[:, -1, :]).abs()     # (B, N) 最后一步
        score = pred_err + recon_err                                  # (B, N)

        all_scores.append(score.cpu().numpy())

    return np.concatenate(all_scores, axis=0)   # (T_windows, N)


# ─────────────────────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: MambGATAD,
    train_loader: DataLoader,
    test_loader: DataLoader,
    test_labels: np.ndarray,
    window_size: int,
    dataset_name: str,
    device: torch.device,
) -> dict:
    """
    标准评估流程（基于对工作区所有论文指标的综合分析）：

    ── 分数计算 ──────────────────────────────────────────────────
    score = |pred_err| + |recon_last_err|    (逐通道，见 collect_scores)

    ── 归一化（单向 z-score）──────────────────────────────────────
    用训练集均值/标准差归一化，clip(0) 保留"高于训练均值"的部分。
    IQR+abs 方案会在异常分数低于训练中位数时反转方向，导致 AUROC<0.5。

    ── 聚合 ───────────────────────────────────────────────────────
    所有通道取均值（全局异常分）。

    ── 指标（依据工作区论文调研）─────────────────────────────────
    1. 标准 AUC-ROC（无 PA，无分数调整）
       - 与 CATCH(ICLR2025)、MTGFlow、CST-GL、MemStream 直接可比
       - 阈值无关，最诚实的评估指标

    2. F1 w/o PA（anomaly-ratio 阈值，无 point adjustment）
       - 与 MSHTrans(KDD2025) Table 1 "w/o PA" 列直接可比

    3. F1 with PA（anomaly-ratio 阈值 + point adjustment）
       - 与 ContrastAD、MSHTrans(KDD2025) "with PA" 列直接可比
       - GDN 也使用 PA，但阈值用验证集最大值而非 anomaly-ratio

    ── 不报告的指标（及原因）────────────────────────────────────
    PA-adjusted 连续 AUROC（即 ContrastAD 代码中的 ts_metrics 实现）：
      point_adjustment 把每段异常内最大分数传播给段内所有点，
      再算 AUROC。在 SMAP（67 段，平均段长 816）上，
      随机模型也能达到 AUROC=0.9988，该指标已完全失去区分力。
    """
    print("\n  [评估] 收集训练集分数...")
    train_scores = collect_scores(model, train_loader, device)   # (T_tr, N)

    print("  [评估] 收集测试集分数...")
    test_scores  = collect_scores(model, test_loader,  device)   # (T_te, N)

    # ── 单向 z-score 归一化 ───────────────────────────────────────
    # 只保留"高于训练均值"的部分；负值（误差低于训练中位数）clip 到 0
    tr_mean = train_scores.mean(axis=0, keepdims=True)
    tr_std  = train_scores.std(axis=0,  keepdims=True) + 1e-4
    z_test  = np.clip((test_scores - tr_mean) / tr_std, 0, None)  # (T_te, N)

    # ── 全通道均值聚合 ────────────────────────────────────────────
    global_score = z_test.mean(axis=1)                             # (T_te,)

    # ── 标签对齐 ──────────────────────────────────────────────────
    T_score = len(global_score)
    offset  = window_size - 1
    label   = test_labels[offset: offset + T_score].astype(int)
    if len(label) < T_score:
        global_score = global_score[:len(label)]
        T_score      = len(label)

    from sklearn.metrics import (roc_auc_score as _auc,
                                  f1_score, precision_score, recall_score)

    # ── 1. 标准 AUC-ROC（无 PA，阈值无关）───────────────────────
    std_auc = float(_auc(label, global_score)) if label.sum() > 0 else 0.0

    # ── 2 & 3. anomaly-ratio 阈值 ─────────────────────────────────
    y_pred_ar = anomaly_ratio_threshold(label, global_score, dataset=dataset_name)

    # 2. F1 w/o PA（MSHTrans Table 1 "w/o PA" 列）
    f1_raw  = float(f1_score(label, y_pred_ar,    zero_division=0))
    prec_raw = float(precision_score(label, y_pred_ar, zero_division=0))
    rec_raw  = float(recall_score(label, y_pred_ar,  zero_division=0))

    # 3. F1 with PA（ContrastAD / MSHTrans "with PA"）
    y_pred_ar_pa = point_adjust(label, y_pred_ar)
    f1_pa   = float(f1_score(label, y_pred_ar_pa,    zero_division=0))
    prec_pa = float(precision_score(label, y_pred_ar_pa, zero_division=0))
    rec_pa  = float(recall_score(label, y_pred_ar_pa,  zero_division=0))

    metrics = {
        # ① 与 CATCH / MTGFlow / CST-GL 直接可比
        "auc_roc":      std_auc,
        # ② 与 MSHTrans w/o PA 直接可比
        "f1_raw":       f1_raw,
        "prec_raw":     prec_raw,
        "rec_raw":      rec_raw,
        # ③ 与 ContrastAD / MSHTrans with PA 直接可比
        "f1_pa_ar":     f1_pa,
        "prec_pa_ar":   prec_pa,
        "rec_pa_ar":    rec_pa,
    }

    W = 58
    print(f"\n{'─'*W}")
    print(f"  数据集: {dataset_name.upper()}")
    print(f"{'─'*W}")
    print(f"  标准 AUC-ROC (无PA，阈值无关)  : {std_auc:.4f}"
          f"  ← 与 CATCH / MTGFlow 直接可比")
    print(f"  F1  w/o PA (anomaly-ratio阈值) : {f1_raw:.4f}"
          f"  P={prec_raw:.4f}  R={rec_raw:.4f}"
          f"  ← 与 MSHTrans(KDD25) w/o PA 直接可比")
    print(f"  F1  with PA (anomaly-ratio阈值): {f1_pa:.4f}"
          f"  P={prec_pa:.4f}  R={rec_pa:.4f}"
          f"  ← 与 ContrastAD / MSHTrans with PA 直接可比")
    print(f"{'─'*W}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    set_seed(cfg["train"]["seed"])

    # ── 设备 ──────────────────────────────────────────────────────
    dev_str = cfg["train"]["device"]
    if dev_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，切换到 CPU")
        dev_str = "cpu"
    if dev_str == "mps" and not torch.backends.mps.is_available():
        print("[WARN] MPS 不可用，切换到 CPU")
        dev_str = "cpu"
    device = torch.device(dev_str)
    print(f"[Info] 设备: {device}")

    # ── 数据 ──────────────────────────────────────────────────────
    window_size  = cfg["data"]["window_size"]
    dataset_name = cfg["data"]["dataset"]

    train_loader, test_loader, test_labels, n_channels = build_loaders(
        data_dir       = cfg["data"]["data_dir"],
        dataset        = dataset_name,
        fmt            = cfg["data"].get("format", "AT").upper(),
        label_file     = cfg["data"].get("label_file"),
        window_size    = window_size,
        train_step     = cfg["data"].get("window_step", 1),
        test_step      = cfg["data"].get("test_step", 1),
        batch_size     = cfg["train"]["batch_size"],
        normalize_data = cfg["data"].get("normalize", True),
        num_workers    = cfg["train"].get("num_workers", 0),
    )

    # ── 验证集（从训练集末尾切 20%）──────────────────────────────
    val_ratio = cfg["train"].get("val_ratio", 0.2)
    train_np  = train_loader.dataset.data.numpy()
    val_cut   = int(len(train_np) * (1.0 - val_ratio))

    train_ds = TimeSeriesDataset(train_np[:val_cut], window_size,
                                  cfg["data"].get("window_step", 1))
    val_arr  = train_np[val_cut:]
    if len(val_arr) <= window_size:
        print("[WARN] 训练集太短，无法切验证集，用全量训练集做早停")
        train_ds = TimeSeriesDataset(train_np, window_size,
                                      cfg["data"].get("window_step", 1))
        val_arr  = train_np
    val_ds = TimeSeriesDataset(val_arr, window_size, step=1)

    train_loader_fit = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=True, num_workers=cfg["train"].get("num_workers", 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=0,
    )
    print(f"[Data] 训练窗口={len(train_ds):,}  验证窗口={len(val_ds):,}  通道数={n_channels}")

    # ── 模型 ──────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model = MambGATAD(
        n_channels  = n_channels,
        window_size = window_size,
        d_model     = model_cfg.get("d_model",  64),
        n_blocks    = model_cfg.get("n_blocks",  2),
        d_state     = model_cfg.get("d_state",  16),
        d_conv      = model_cfg.get("d_conv",    4),
        expand      = model_cfg.get("expand",    2),
        pred_len    = model_cfg.get("pred_len",  1),
        dropout     = model_cfg.get("dropout", 0.1),
        # v1+ 参数（v0 忽略）
        n_heads      = model_cfg.get("n_heads",       4),
        top_k        = model_cfg.get("top_k",      None),
        # v3+ 参数（v0/v1/v2 忽略）
        patch_sizes  = tuple(model_cfg.get("patch_sizes", [1])),
        n_snapshots  = model_cfg.get("n_snapshots",   4),
    ).to(device)

    print(f"[Model] {model.__class__.__name__} | 参数量: {model.count_parameters():,}")

    # ── 优化器 + 调度器 ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["train"]["lr"],
        weight_decay = cfg["train"].get("weight_decay", 1e-4),
    )
    n_epochs = cfg["train"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6,
    )

    loss_cfg  = cfg.get("loss", {})
    criterion = AnomalyLoss(
        beta    = loss_cfg.get("beta",    0.5),
        lambda1 = loss_cfg.get("lambda1", 0.0),   # v2+: 频域损失权重
        lambda2 = loss_cfg.get("lambda2", 0.0),   # v2+: 形状损失权重
    )

    # ── Checkpoint ────────────────────────────────────────────────
    save_dir  = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    # 版本号优先级：CLI --version > config version > models.VERSION 常量
    if "version" not in cfg:
        cfg["version"] = getattr(_models_pkg, "VERSION", "v0")
    version = cfg["version"]
    print(f"[Info] 实验版本: {version}")
    best_path = save_dir / f"best_{dataset_name}_{version}.pt"
    last_path = save_dir / f"last_{dataset_name}_{version}.pt"

    best_val_loss = float("inf")
    patience_cnt  = 0
    patience      = cfg["train"].get("patience", 10)

    print(f"\n{'═'*55}")
    print(f"  训练开始  |  {dataset_name.upper()}  |  {n_epochs} epochs")
    print(f"{'═'*55}")

    for epoch in range(1, n_epochs + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        train_losses = []
        t0 = time.time()

        for x_batch, y_batch in tqdm(
            train_loader_fit,
            desc=f"Epoch {epoch:03d}/{n_epochs}",
            leave=False, ncols=75,
        ):
            x_batch = x_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            pred, recon, _, aux = model(x_batch)
            loss = criterion(pred, y_batch, recon=recon, x=x_batch, aux_loss=aux)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        avg_train = float(np.mean(train_losses))

        # ── Validation ─────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v = x_v.to(device, dtype=torch.float32)
                y_v = y_v.to(device, dtype=torch.float32)
                pred_v, recon_v, _, aux_v = model(x_v)
                val_losses.append(
                    criterion(pred_v, y_v, recon=recon_v, x=x_v, aux_loss=aux_v).item()
                )
        avg_val = float(np.mean(val_losses))

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:03d}  train={avg_train:.5f}  val={avg_val:.5f}"
              f"  lr={scheduler.get_last_lr()[0]:.1e}  {elapsed:.1f}s")

        # ── 早停 + 保存 ────────────────────────────────────────
        if np.isnan(avg_train) or np.isnan(avg_val):
            print("  [WARN] loss=nan，跳过保存")
            continue

        ckpt = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "val_loss":   avg_val,
            "cfg":        cfg,
            "n_channels": n_channels,
        }
        torch.save(ckpt, last_path)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_cnt  = 0
            torch.save(ckpt, best_path)
            print(f"  ✓ 最佳模型已保存 (val={avg_val:.5f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n  早停（{patience} 轮无改善）")
                break

    # ── 最终评估 ──────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  加载最佳模型进行评估...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    # 用完整训练集收集分数（不切验证集）
    full_train_loader, _, _, _ = build_loaders(
        data_dir       = cfg["data"]["data_dir"],
        dataset        = dataset_name,
        fmt            = cfg["data"].get("format", "AT").upper(),
        label_file     = cfg["data"].get("label_file"),
        window_size    = window_size,
        train_step     = cfg["data"].get("window_step", 1),
        test_step      = cfg["data"].get("test_step", 1),
        batch_size     = cfg["train"]["batch_size"],
        normalize_data = cfg["data"].get("normalize", True),
        num_workers    = 0,
    )

    metrics = evaluate(
        model        = model,
        train_loader = full_train_loader,
        test_loader  = test_loader,
        test_labels  = test_labels,
        window_size  = window_size,
        dataset_name = dataset_name,
        device       = device,
    )

    # 保存结果
    result_path = save_dir / f"results_{dataset_name}_{version}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({k: round(float(v), 6) for k, v in metrics.items()}, f, indent=2)
    print(f"  结果已保存 → {result_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="config/smap.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--epochs",  default=None, type=int)
    p.add_argument("--device",  default=None)
    p.add_argument("--version", default=None,
                   help="实验版本号，用于区分 checkpoint 和结果文件（如 v0/v1/v2）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    if args.dataset: cfg["data"]["dataset"]   = args.dataset
    if args.epochs:  cfg["train"]["epochs"]   = args.epochs
    if args.device:  cfg["train"]["device"]   = args.device
    if args.version: cfg["version"]           = args.version
    train(cfg)
