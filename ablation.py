"""
ablation.py — 系统消融实验

逐一测试每个改进组件的贡献，每个配置同时报：
  - AUC(recon): 重建/预测误差评分
  - AUC(repr):  encoder 表示空间 Mahalanobis 距离评分

配置从简到繁，每次只加一个组件：
  ① 纯预测损失（最简 baseline）
  ② +重建损失
  ③ +多尺度 Patch 嵌入
  ④ +频域损失（freq + shape）
  ⑤ 完整 v2（+所有改进，对比损失已关闭）

用法：
  python ablation.py --data_dir ./datasets/AT/SMAP --epochs 5
  python ablation.py --data_dir ./datasets/AT/MSL  --epochs 5
  python ablation.py --data_dir ./datasets/AT/SMD  --epochs 5 --dataset smd
"""

import argparse
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, ".")
from data.dataset import load_at_format, TimeSeriesDataset
from models.mambgat import MambGATAD, PredictionLoss

# ─────────────────────────────────────────────────────────────────────────────
# 消融配置表（从简到繁，每次只加一个组件）
# ─────────────────────────────────────────────────────────────────────────────
ABLATION_CONFIGS = [
    {
        "name":  "① 纯预测",
        "desc":  "单点Patch + 仅预测损失（最简baseline）",
        "patch_sizes": (1,),
        "loss":  {"alpha": 0.5, "beta": 0.0, "lambda1": 0.0, "lambda2": 0.0, "lambda_c": 0.0},
    },
    {
        "name":  "② +重建损失",
        "desc":  "单点Patch + 预测损失 + 重建损失",
        "patch_sizes": (1,),
        "loss":  {"alpha": 0.5, "beta": 0.1, "lambda1": 0.0, "lambda2": 0.0, "lambda_c": 0.0},
    },
    {
        "name":  "③ +多尺度Patch",
        "desc":  "多尺度Patch(4,8,16) + 预测 + 重建",
        "patch_sizes": (4, 8, 16),
        "loss":  {"alpha": 0.5, "beta": 0.1, "lambda1": 0.0, "lambda2": 0.0, "lambda_c": 0.0},
    },
    {
        "name":  "④ +频域损失",
        "desc":  "多尺度Patch + 预测 + 重建 + 频域 + 形状",
        "patch_sizes": (4, 8, 16),
        "loss":  {"alpha": 0.5, "beta": 0.1, "lambda1": 0.1, "lambda2": 0.05, "lambda_c": 0.0},
    },
]

# 推理超参（比正式训练小，加速消融）
WINDOW     = 100
D_MODEL    = 64    # 正式 128，这里减半加速
N_BLOCKS   = 2
N_HEADS    = 4
BATCH      = 256
TRAIN_STEP = 5
LR         = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# 评分工具函数
# ─────────────────────────────────────────────────────────────────────────────

def topk_mean_agg(z: np.ndarray, k: int = 3) -> np.ndarray:
    k = min(k, z.shape[1])
    return np.partition(z, -k, axis=1)[:, -k:].mean(axis=1)


def collect_recon(model, loader, device, desc=""):
    """收集重建+预测误差，返回 (T, N)"""
    model.eval()
    out = []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=f"    {desc}", ncols=72, leave=False):
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.float32)
            pred, recon, _, ___ = model(xb)
            pe = (pred.squeeze(-1) - yb).abs()
            re = (recon - xb).abs().mean(dim=1)
            out.append((pe + re).cpu().numpy())
    return np.concatenate(out, axis=0)


def collect_repr(model, loader, device, desc=""):
    """收集 encoder 最后时间步表示，返回 (T, N, D)"""
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in tqdm(loader, desc=f"    {desc}", ncols=72, leave=False):
            xb = xb.to(device, dtype=torch.float32)
            out.append(model.encode(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_auc(scores_1d, test_labels, label_offset):
    n = min(len(scores_1d), len(test_labels) - label_offset)
    try:
        return roc_auc_score(
            test_labels[label_offset:label_offset + n].astype(int),
            scores_1d[:n]
        )
    except Exception:
        return float("nan")


def eval_both(model, tr_loader, te_loader, test_labels, device, label_offset):
    """同时计算 recon-AUC 和 repr-AUC，返回 (auc_recon, auc_repr)"""
    N = model.n_channels

    # ── recon 评分 ──────────────────────────────────────────────
    tr_sc = collect_recon(model, tr_loader, device, "recon(train)")
    te_sc = collect_recon(model, te_loader, device, "recon(test) ")
    tr_med = np.median(tr_sc, axis=0, keepdims=True)
    tr_iqr = (np.percentile(tr_sc, 75, axis=0, keepdims=True)
              - np.percentile(tr_sc, 25, axis=0, keepdims=True) + 0.01)
    z_te = np.abs(te_sc - tr_med) / tr_iqr
    auc_recon = compute_auc(topk_mean_agg(z_te, k=min(3, N)),
                             test_labels, label_offset)

    # ── repr 评分 ───────────────────────────────────────────────
    tr_rp = collect_repr(model, tr_loader, device, "repr (train)")
    te_rp = collect_repr(model, te_loader, device, "repr (test) ")
    D = tr_rp.shape[2]
    scores_repr = np.zeros((len(te_rp), N))
    for n in range(N):
        mu  = tr_rp[:, n, :].mean(0)
        std = tr_rp[:, n, :].std(0) + 1e-6
        scores_repr[:, n] = np.linalg.norm(
            (te_rp[:, n, :] - mu) / std, axis=1)
    auc_repr = compute_auc(topk_mean_agg(scores_repr, k=min(3, N)),
                            test_labels, label_offset)

    return auc_recon, auc_repr


# ─────────────────────────────────────────────────────────────────────────────
# 单个配置的训练 + 评估
# ─────────────────────────────────────────────────────────────────────────────

def run_config(cfg: dict, n_channels: int,
               tr_loader, te_loader, test_labels: np.ndarray,
               n_epochs: int, device: torch.device) -> dict:

    torch.manual_seed(42); np.random.seed(42)
    tag = cfg["name"]

    model = MambGATAD(
        n_channels  = n_channels,
        window_size = WINDOW,
        d_model     = D_MODEL,
        n_blocks    = N_BLOCKS,
        n_heads     = N_HEADS,
        d_state     = 16,
        d_conv      = 4,
        expand      = 2,
        pred_len    = 1,
        dropout     = 0.1,
        patch_sizes = cfg["patch_sizes"],
        n_snapshots = 4,
    ).to(device)

    criterion = PredictionLoss(**cfg["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6)

    label_offset = WINDOW - 1
    history = []

    print(f"\n  {tag}  —  {cfg['desc']}")
    epoch_bar = tqdm(range(1, n_epochs + 1), desc=f"  {tag}",
                     ncols=72, position=0)

    for epoch in epoch_bar:
        t0 = time.time()
        model.train()
        losses = []
        batch_bar = tqdm(tr_loader, desc=f"    训练", ncols=72,
                         leave=False, position=1)
        for xb, yb in batch_bar:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            pred, recon, _, cl = model(xb)
            loss = criterion(pred, yb, recon=recon, x=xb, contrast_loss=cl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        avg_loss = float(np.mean(losses))

        model.eval()
        auc_recon, auc_repr = eval_both(
            model, tr_loader, te_loader, test_labels, device, label_offset)

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "loss": avg_loss,
                         "auc_recon": auc_recon, "auc_repr": auc_repr})
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}",
                               recon=f"{auc_recon:.4f}",
                               repr=f"{auc_repr:.4f}")
        tqdm.write(f"  {tag}  Epoch {epoch:02d}/{n_epochs}  "
                   f"loss={avg_loss:.5f}  "
                   f"AUC(recon)={auc_recon:.4f}  "
                   f"AUC(repr)={auc_repr:.4f}  ({elapsed:.0f}s)")

    epoch_bar.close()
    return {"config": cfg["name"], "history": history,
            "best_recon": max(h["auc_recon"] for h in history),
            "best_repr":  max(h["auc_repr"]  for h in history),
            "last_loss":  history[-1]["loss"]}


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./datasets/AT/SMAP")
    parser.add_argument("--dataset",  default="smap",
                        choices=["smap", "msl", "smd", "psm", "swat"])
    parser.add_argument("--epochs",   type=int, default=5)
    parser.add_argument("--device",   default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    print("=" * 65)
    print(f"  MambGAT-AD 消融实验  |  数据集={args.dataset.upper()}")
    print(f"  epochs={args.epochs}  device={device}")
    print("=" * 65)

    # ── 数据 ────────────────────────────────────────────────────
    print(f"\n[Data] 加载 {args.dataset.upper()} 数据...")
    train_np, test_np, test_labels = load_at_format(
        args.data_dir, dataset=args.dataset, normalize=True)
    n_channels = train_np.shape[1]

    tr_ds = TimeSeriesDataset(train_np, WINDOW, step=TRAIN_STEP)
    te_ds = TimeSeriesDataset(test_np,  WINDOW, step=1)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"[Data] 通道={n_channels}  训练窗口={len(tr_ds):,}  "
          f"测试窗口={len(te_ds):,}  异常率={test_labels.mean():.2%}")

    # ── 逐配置运行 ───────────────────────────────────────────────
    results = []
    for cfg in ABLATION_CONFIGS:
        res = run_config(cfg, n_channels, tr_loader, te_loader,
                         test_labels, args.epochs, device)
        results.append(res)

    # ── 汇总表 ──────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print(f"  消融结果汇总  |  {args.dataset.upper()}")
    print(f"  每列取 {args.epochs} epoch 内最佳 AUC")
    print("=" * 65)
    print(f"  {'配置':<18}  {'AUC(recon)':>10}  {'AUC(repr)':>10}  {'repr提升':>8}")
    print(f"  {'─'*55}")

    base_repr  = results[0]["best_repr"]
    base_recon = results[0]["best_recon"]
    for r in results:
        delta = r["best_repr"] - base_repr
        marker = f"+{delta:.4f} ✓" if delta > 0.005 else (
                 f"{delta:+.4f} ✗" if delta < -0.005 else f"{delta:+.4f} ─")
        print(f"  {r['config']:<18}  {r['best_recon']:>10.4f}  "
              f"{r['best_repr']:>10.4f}  {marker:>12}")

    print(f"\n  结论：")
    best = max(results, key=lambda r: r["best_repr"])
    print(f"  repr 最佳配置: {best['config']}  AUC={best['best_repr']:.4f}")
    best_r = max(results, key=lambda r: r["best_recon"])
    print(f"  recon 最佳配置: {best_r['config']}  AUC={best_r['best_recon']:.4f}")

    # 判断哪些改进有效
    print(f"\n  各组件效果（对比上一步的 repr AUC）：")
    for i in range(1, len(results)):
        prev = results[i-1]["best_repr"]
        curr = results[i]["best_repr"]
        delta = curr - prev
        added = ABLATION_CONFIGS[i]["name"]
        if delta > 0.01:
            print(f"  {added}: ✅ +{delta:.4f}")
        elif delta < -0.01:
            print(f"  {added}: ❌ {delta:.4f}（有害）")
        else:
            print(f"  {added}: ➖ {delta:+.4f}（无显著效果）")

    print("=" * 65)


if __name__ == "__main__":
    main()
