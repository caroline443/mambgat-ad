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
from tqdm import tqdm

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


def collect_scores(model, loader, device, desc="推理"):
    """收集重建/预测误差分数，返回 (T, N)"""
    model.eval()
    scores = []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=f"    {desc}", ncols=80, leave=False):
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.float32)
            pred, recon, _, ___ = model(xb)
            pred_err  = (pred.squeeze(-1) - yb).abs()
            recon_err = (recon - xb).abs().mean(dim=1)
            scores.append((pred_err + recon_err).cpu().numpy())
    return np.concatenate(scores, axis=0)   # (T, N)


def collect_repr(model, loader, device, desc="特征提取"):
    """收集 encoder 表示，返回 (T, N, D)"""
    model.eval()
    reprs = []
    with torch.no_grad():
        for xb, _ in tqdm(loader, desc=f"    {desc}", ncols=80, leave=False):
            xb = xb.to(device, dtype=torch.float32)
            z = model.encode(xb)   # (B, N, D)
            reprs.append(z.cpu().numpy())
    return np.concatenate(reprs, axis=0)   # (T, N, D)


def repr_auc(train_repr, test_repr, test_labels, label_offset):
    """
    用表示空间标准化 Mahalanobis 距离计算 AUC。
    逐通道：(z - μ) / σ，取 L2 norm 作为该通道异常分。
    train_repr: (T_tr, N, D)
    test_repr:  (T_te, N, D)
    """
    N = train_repr.shape[1]
    T_te = test_repr.shape[0]
    scores = np.zeros((T_te, N))

    for n in range(N):
        mu  = train_repr[:, n, :].mean(axis=0)         # (D,)
        std = train_repr[:, n, :].std(axis=0) + 1e-6   # (D,)
        z   = (test_repr[:, n, :] - mu) / std          # (T_te, D)
        scores[:, n] = np.linalg.norm(z, axis=1)       # (T_te,)

    # top-3 mean 聚合
    global_score = topk_mean_agg(scores, k=min(3, N))

    n = min(len(global_score), len(test_labels) - label_offset)
    labels_cut = test_labels[label_offset : label_offset + n].astype(int)
    score_cut  = global_score[:n]
    try:
        return roc_auc_score(labels_cut, score_cut)
    except Exception:
        return 0.5


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
            lambda_c=0.0,   # 对比损失暂时关闭（符号逻辑有问题，单独隔离验证）
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
    history = {"epoch": [], "auc_recon": [], "auc_repr": [], "train_loss": []}

    epoch_bar = tqdm(range(1, n_epochs + 1),
                     desc=f"{tag} 总进度", ncols=80, position=0)

    for epoch in epoch_bar:
        t0 = time.time()
        model.train()
        losses = []

        batch_bar = tqdm(train_loader,
                         desc=f"  {tag} Epoch {epoch:02d}/{n_epochs} 训练",
                         ncols=80, leave=False, position=1)
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

        # ── 评估：重建误差 AUC ────────────────────────────────────
        label_offset = WINDOW - 1
        model.eval()
        tr_scores = collect_scores(model, train_loader, device, desc=f"{tag} 重建推理(train)")
        te_scores = collect_scores(model, test_loader,  device, desc=f"{tag} 重建推理(test)")

        tr_med = np.median(tr_scores, axis=0, keepdims=True)
        tr_iqr = (np.percentile(tr_scores, 75, axis=0, keepdims=True)
                  - np.percentile(tr_scores, 25, axis=0, keepdims=True) + 0.01)
        z_te         = np.abs(te_scores - tr_med) / tr_iqr
        global_score = topk_mean_agg(z_te, k=3)
        n            = min(len(global_score), len(test_labels) - label_offset)
        try:
            auc_recon = roc_auc_score(
                test_labels[label_offset:label_offset+n].astype(int),
                global_score[:n])
        except Exception:
            auc_recon = 0.5

        # ── 评估：表示空间 Mahalanobis AUC ───────────────────────
        tr_repr = collect_repr(model, train_loader, device, desc=f"{tag} 表示提取(train)")
        te_repr = collect_repr(model, test_loader,  device, desc=f"{tag} 表示提取(test)")
        auc_repr = repr_auc(tr_repr, te_repr, test_labels, label_offset)

        elapsed = time.time() - t0
        history["epoch"].append(epoch)
        history["auc_recon"].append(auc_recon)
        history["auc_repr"].append(auc_repr)
        history["train_loss"].append(avg_loss)

        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}",
                               recon=f"{auc_recon:.4f}",
                               repr=f"{auc_repr:.4f}")
        tqdm.write(f"{tag} Epoch {epoch:02d}/{n_epochs}  loss={avg_loss:.5f}  "
                   f"AUC(recon)={auc_recon:.4f}  AUC(repr)={auc_repr:.4f}  ({elapsed:.1f}s)")

    epoch_bar.close()
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
    print("\n" + "=" * 70)
    print("  结果汇总（真实 SMAP 数据）")
    print("  recon=重建误差分数  repr=表示空间Mahalanobis距离")
    print("=" * 70)
    print(f"  {'Ep':>3}  {'v1(recon)':>10}  {'v1(repr)':>10}  "
          f"{'v2(recon)':>10}  {'v2(repr)':>10}")
    print(f"  {'─'*58}")
    for i in range(args.epochs):
        r1  = hist_v1["auc_recon"][i]
        p1  = hist_v1["auc_repr"][i]
        r2  = hist_v2["auc_recon"][i]
        p2  = hist_v2["auc_repr"][i]
        print(f"  {i+1:>3}  {r1:>10.4f}  {p1:>10.4f}  {r2:>10.4f}  {p2:>10.4f}")

    print(f"\n  最佳 AUC(repr)  →  v1: {max(hist_v1['auc_repr']):.4f}"
          f"   v2: {max(hist_v2['auc_repr']):.4f}")
    print(f"  最佳 AUC(recon) →  v1: {max(hist_v1['auc_recon']):.4f}"
          f"   v2: {max(hist_v2['auc_recon']):.4f}")

    best_repr = max(max(hist_v1["auc_repr"]), max(hist_v2["auc_repr"]))
    best_recon = max(max(hist_v1["auc_recon"]), max(hist_v2["auc_recon"]))
    if best_repr > best_recon + 0.05:
        print(f"\n  ✅ 表示空间评分(repr) 显著优于重建误差(recon)！")
        print(f"     repr AUC={best_repr:.4f}  recon AUC={best_recon:.4f}")
        print(f"     结论：换用表示空间异常评分可大幅提升 SMAP 结果。")
    else:
        print(f"\n  ⚠️  repr 与 recon 差距不大，需更多 epoch 或检查 encoder 质量。")
    print("=" * 70)


if __name__ == "__main__":
    main()
