"""
fast_compare.py — 用真实 SMAP 数据快速对比 v1 vs v2

策略：
  - 直接读 datasets/AT/SMAP/ 下的真实数据
  - v1：patch_sizes=(1,)，关闭频域/对比损失（退化为原始行为）
  - v2：patch_sizes=(4,8,16)，开启全部三项改进
  - 各跑 N_EPOCHS 轮，每轮输出 AUC-ROC
  - 最后打印对比表，判断改进是否有效

用法：
  python fast_compare.py                          # 默认 5 epoch，CUDA
  python fast_compare.py --epochs 10 --device cpu
  python fast_compare.py --data_dir path/to/SMAP  # 自定义数据路径
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

sys.path.insert(0, ".")
from data.dataset import load_at_format, TimeSeriesDataset
from models.mambgat import MambGATAD, PredictionLoss

# ─────────────────────────────────────────────────────────────────────────────
# 默认超参（比正式训练小，加速验证）
# ─────────────────────────────────────────────────────────────────────────────
WINDOW      = 100
D_MODEL     = 64      # 正式 128，这里减半
N_BLOCKS    = 2       # 正式 3
N_HEADS     = 4       # 正式 8
BATCH       = 256     # 大 batch 加速 GPU
TRAIN_STEP  = 5       # 训练集滑窗步长（减少窗口数，加速）
LR          = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def topk_mean_agg(z: np.ndarray, k: int = 3) -> np.ndarray:
    k = min(k, z.shape[1])
    topk = np.partition(z, -k, axis=1)[:, -k:]
    return topk.mean(axis=1)


def collect_scores(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, dtype=torch.float32)
            _, __, sc, ___ = model(xb)
            scores.append(sc.cpu().numpy())
    return np.concatenate(scores, axis=0)   # (T, N)


# ─────────────────────────────────────────────────────────────────────────────
# 单次实验
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    version: str,
    n_channels: int,
    train_loader,
    test_loader,
    test_labels: np.ndarray,
    train_data_np: np.ndarray,
    n_epochs: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)

    # ── 构建模型 ──────────────────────────────────────────────────
    if version == "v1":
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
            patch_sizes = (1,),     # 退化为单点投影
            n_snapshots = 4,
        ).to(device)
        criterion = PredictionLoss(
            alpha=0.5, beta=0.1,
            lambda1=0.0,    # 关闭频域损失
            lambda2=0.0,    # 关闭形状损失
            lambda_c=0.0,   # 关闭图对比
        )
    else:
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
            patch_sizes = (4, 8, 16),
            n_snapshots = 4,
        ).to(device)
        criterion = PredictionLoss(
            alpha=0.5, beta=0.1,
            lambda1=0.1,
            lambda2=0.05,
            lambda_c=-0.4,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    tag = f"[{version.upper()}]"
    print(f"\n{tag} 参数量: {model.count_parameters():,}")
    print(f"{tag} {'Epoch':>5}  {'TrainLoss':>10}  {'AUC-ROC':>8}  {'Time':>6}")
    print(f"{tag} {'─'*42}")

    # 预先计算训练集 IQR 统计量（用于 AUC 评估）
    # 先用随机初始化的模型跑一遍，后续每 epoch 更新
    history = {"epoch": [], "auc": [], "train_loss": []}

    # 构建训练集 loader（用于收集 IQR 基准，step=1 更准确但慢，这里复用 train_loader）
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            pred, recon, _, cl = model(xb)
            loss = criterion(pred, yb, recon=recon, x=xb, contrast_loss=cl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        avg_loss = float(np.mean(losses))

        # ── 评估 AUC ──────────────────────────────────────────────
        model.eval()
        tr_scores = collect_scores(model, train_loader, device)   # (T_tr, N)
        te_scores = collect_scores(model, test_loader,  device)   # (T_te, N)

        # IQR 归一化（GDN 风格）
        tr_med = np.median(tr_scores, axis=0, keepdims=True)
        tr_iqr = (np.percentile(tr_scores, 75, axis=0, keepdims=True)
                  - np.percentile(tr_scores, 25, axis=0, keepdims=True) + 0.01)
        z_te = np.abs(te_scores - tr_med) / tr_iqr

        # top-3 mean 聚合
        global_score = topk_mean_agg(z_te, k=3)

        # 标签对齐（窗口末尾对应 label[i + W - 1]）
        label_offset = WINDOW - 1
        n = min(len(global_score), len(test_labels) - label_offset)
        labels_cut = test_labels[label_offset : label_offset + n].astype(int)
        score_cut  = global_score[:n]

        try:
            auc = roc_auc_score(labels_cut, score_cut)
        except Exception:
            auc = 0.5

        elapsed = time.time() - t0
        history["epoch"].append(epoch)
        history["auc"].append(auc)
        history["train_loss"].append(avg_loss)

        print(f"{tag} {epoch:>5}  {avg_loss:>10.5f}  {auc:>8.4f}  {elapsed:>5.1f}s")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./datasets/AT/SMAP", help="SMAP 数据目录")
    parser.add_argument("--epochs",   type=int, default=5,           help="训练轮数")
    parser.add_argument("--device",   default="cuda",                help="cuda / cpu")
    args = parser.parse_args()

    # 设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，自动切换到 CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    print("=" * 60)
    print("  MambGAT-AD 快速对比：v1 vs v2（真实 SMAP 数据）")
    print(f"  epochs={args.epochs}  device={device}")
    print(f"  data_dir={args.data_dir}")
    print("=" * 60)

    # ── 加载数据 ──────────────────────────────────────────────────
    print("\n[Data] 加载 SMAP 数据...")
    train_np, test_np, test_labels = load_at_format(
        args.data_dir, dataset="smap", normalize=True
    )
    n_channels = train_np.shape[1]

    # 构建 DataLoader
    train_ds = TimeSeriesDataset(train_np, WINDOW, step=TRAIN_STEP)
    test_ds  = TimeSeriesDataset(test_np,  WINDOW, step=1)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                              num_workers=0)

    print(f"[Data] 训练窗口={len(train_ds):,}  测试窗口={len(test_ds):,}")

    # ── 运行 v1 ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  v1：单点投影 + 纯 MSE 损失（原始行为）")
    print("─" * 60)
    hist_v1 = run_experiment(
        "v1", n_channels, train_loader, test_loader, test_labels,
        train_np, args.epochs, device,
    )

    # ── 运行 v2 ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  v2：多尺度Patch + 频域损失 + 图对比正则化")
    print("─" * 60)
    hist_v2 = run_experiment(
        "v2", n_channels, train_loader, test_loader, test_labels,
        train_np, args.epochs, device,
    )

    # ── 汇总 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  结果汇总（真实 SMAP 数据）")
    print("=" * 60)
    print(f"  {'Epoch':>5}  {'v1 AUC':>8}  {'v2 AUC':>8}  {'Δ AUC':>8}")
    print(f"  {'─'*40}")
    for i in range(args.epochs):
        a1 = hist_v1["auc"][i]
        a2 = hist_v2["auc"][i]
        delta = a2 - a1
        marker = " ✓" if delta > 0 else " ✗"
        print(f"  {i+1:>5}  {a1:>8.4f}  {a2:>8.4f}  {delta:>+8.4f}{marker}")

    best_v1 = max(hist_v1["auc"])
    best_v2 = max(hist_v2["auc"])
    last_v1 = hist_v1["auc"][-1]
    last_v2 = hist_v2["auc"][-1]

    print(f"\n  最佳 AUC  →  v1: {best_v1:.4f}   v2: {best_v2:.4f}   Δ={best_v2-best_v1:+.4f}")
    print(f"  末轮 AUC  →  v1: {last_v1:.4f}   v2: {last_v2:.4f}   Δ={last_v2-last_v1:+.4f}")

    if best_v2 > best_v1 + 0.01:
        print(f"\n  ✅ v2 改进有效！AUC 提升 {best_v2-best_v1:+.4f}")
        print("     建议：去服务器用 config/smap.yaml 全量训练（50 epoch）。")
    elif best_v2 >= best_v1 - 0.01:
        print(f"\n  ⚠️  v2 与 v1 接近（差距 < 0.01），5 epoch 可能还未收敛。")
        print("     建议：加到 --epochs 15 再跑一次，或直接全量训练。")
    else:
        print(f"\n  ❌ v2 在 5 epoch 内未超过 v1，建议检查损失权重或增大 epoch。")

    print("=" * 60)


if __name__ == "__main__":
    main()
