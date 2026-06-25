"""
quick_validate.py — 快速验证 v1 vs v2 改进效果

策略：
  - 用合成数据（多通道正弦波 + 注入异常段）模拟 SMAP 分布
  - 分别训练 v1（原始模型）和 v2（三项改进）各 N_EPOCHS 轮
  - 对比 AUC-ROC 变化曲线，验证改进方向是否有效
  - 全程在 CPU 上跑，无需 GPU / 真实数据集

用法：
  python quick_validate.py
  python quick_validate.py --epochs 10 --n_channels 25 --seq_len 3000
"""

import argparse
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, ".")
from models.mambgat import MambGATAD, PredictionLoss

# ─────────────────────────────────────────────────────────────────────────────
# 超参（快速验证用，比正式训练小很多）
# ─────────────────────────────────────────────────────────────────────────────
WINDOW   = 100      # 滑动窗口长度（与 SMAP 一致）
D_MODEL  = 64       # 嵌入维度（正式 128，这里减半加速）
N_BLOCKS = 2        # 编码器块数（正式 3）
N_HEADS  = 4        # 注意力头数
BATCH    = 64
LR       = 1e-3
DEVICE   = torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 合成数据生成
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_data(
    n_channels: int = 25,
    train_len:  int = 5000,
    test_len:   int = 8000,
    anomaly_ratio: float = 0.13,   # SMAP 异常率 13.13%
    seed: int = 42,
) -> tuple:
    """
    生成多通道正弦波时序数据，并在测试集中注入异常段。

    正常模式：每个通道是不同频率/相位的正弦波叠加（模拟遥测周期性）
    异常模式：
      - 幅值突变（spike）：某段幅值乘以 3~5 倍
      - 频率漂移（drift）：某段频率偏移 50%
      - 相关性破坏（coupling）：某段通道间相关性被打乱

    返回：
      train_data: (train_len, n_channels) float32
      test_data:  (test_len,  n_channels) float32
      test_labels:(test_len,) int  0/1
    """
    rng = np.random.default_rng(seed)

    # 每个通道的基础频率和相位
    freqs  = rng.uniform(0.01, 0.05, n_channels)
    phases = rng.uniform(0, 2 * np.pi, n_channels)
    amps   = rng.uniform(0.5, 1.5, n_channels)

    def make_signal(length, inject_anomaly=False):
        t = np.arange(length)
        # 基础信号：正弦 + 少量噪声
        sig = np.stack([
            amps[i] * np.sin(2 * np.pi * freqs[i] * t + phases[i])
            for i in range(n_channels)
        ], axis=1)  # (length, n_channels)
        sig += rng.normal(0, 0.05, sig.shape)

        labels = np.zeros(length, dtype=int)

        if not inject_anomaly:
            return sig.astype(np.float32), labels

        # 注入异常段（总异常率 ≈ anomaly_ratio）
        n_anomaly_pts = int(length * anomaly_ratio)
        # 分成 5~10 个异常段
        n_segs = rng.integers(5, 10)
        seg_len = n_anomaly_pts // n_segs

        for _ in range(n_segs):
            start = rng.integers(WINDOW, length - seg_len - 1)
            end   = start + seg_len
            atype = rng.integers(0, 3)

            if atype == 0:
                # 幅值突变：随机几个通道乘以 3~5
                ch = rng.choice(n_channels, size=rng.integers(1, 5), replace=False)
                sig[start:end, ch] *= rng.uniform(3, 5)
            elif atype == 1:
                # 频率漂移：重新生成该段信号（频率偏移 50%）
                t_seg = np.arange(seg_len)
                for i in range(n_channels):
                    sig[start:end, i] = amps[i] * np.sin(
                        2 * np.pi * freqs[i] * 1.5 * t_seg + phases[i]
                    ) + rng.normal(0, 0.05, seg_len)
            else:
                # 相关性破坏：通道间随机打乱
                perm = rng.permutation(n_channels)
                sig[start:end] = sig[start:end][:, perm]

            labels[start:end] = 1

        return sig.astype(np.float32), labels

    train_data, _           = make_signal(train_len, inject_anomaly=False)
    test_data,  test_labels = make_signal(test_len,  inject_anomaly=True)

    # 归一化（用训练集统计量）
    mu  = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True) + 1e-6
    train_data = (train_data - mu) / std
    test_data  = (test_data  - mu) / std

    return train_data, test_data, test_labels


def make_windows(data: np.ndarray, window: int, step: int = 1):
    """把 (T, N) 切成 (n_windows, window, N) 的滑动窗口"""
    T, N = data.shape
    xs, ys = [], []
    for i in range(0, T - window, step):
        xs.append(data[i : i + window])
        ys.append(data[i + window])       # 预测下一步
    return np.stack(xs), np.stack(ys)


# ─────────────────────────────────────────────────────────────────────────────
# 单次训练 + 评估
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    model_v: str,           # "v1" 或 "v2"
    n_channels: int,
    train_data: np.ndarray,
    test_data:  np.ndarray,
    test_labels: np.ndarray,
    n_epochs: int,
    verbose: bool = True,
) -> dict:
    """
    训练一个模型版本，每 epoch 记录 AUC，返回结果字典。
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # ── 构建数据集 ──────────────────────────────────────────────
    step = 5   # 快速验证用 step=5，减少窗口数
    X_tr, Y_tr = make_windows(train_data, WINDOW, step=step)
    X_te, Y_te = make_windows(test_data,  WINDOW, step=1)

    # 测试集标签对齐（窗口末尾对应的标签）
    # score[i] 基于 pred(data[i+W])，对齐 label[i+W]，偏移 = WINDOW
    label_offset = WINDOW
    te_labels_aligned = test_labels[WINDOW : WINDOW + len(X_te)]
    if len(te_labels_aligned) < len(X_te):
        X_te = X_te[:len(te_labels_aligned)]
        Y_te = Y_te[:len(te_labels_aligned)]

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr)),
        batch_size=BATCH, shuffle=True, drop_last=False,
    )
    te_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(Y_te)),
        batch_size=BATCH, shuffle=False,
    )

    # ── 构建模型 ──────────────────────────────────────────────
    if model_v == "v1":
        # v1：用 patch_sizes=(1,) 退化为单点投影，关闭频域/对比损失
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
            patch_sizes = (1,),    # 退化为单点投影（v1 行为）
            n_snapshots = 4,
        ).to(DEVICE)
        criterion = PredictionLoss(
            alpha=0.5, beta=0.1,
            lambda1=0.0,   # 关闭频域损失
            lambda2=0.0,   # 关闭形状损失
            lambda_c=0.0,  # 关闭图对比
        )
    else:
        # v2：完整三项改进
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
        ).to(DEVICE)
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

    # ── 分数收集（用于 IQR 归一化 + AUC 评估）──────────────────
    def collect_scores(loader, desc="推理"):
        model.eval()
        scores = []
        with torch.no_grad():
            for xb, yb in tqdm(loader, desc=f"    {desc}", ncols=80, leave=False):
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred, recon, _, ___ = model(xb)
                pred_err  = (pred.squeeze(-1) - yb).abs()
                recon_err = (recon - xb).abs().mean(dim=1)
                scores.append((pred_err + recon_err).cpu().numpy())
        return np.concatenate(scores, axis=0)  # (T, N)

    history = {"epoch": [], "auc": [], "train_loss": []}
    tag = f"[{model_v.upper()}]"

    if verbose:
        print(f"\n{tag} 参数量: {model.count_parameters():,}  "
              f"训练窗口: {len(X_tr):,}  测试窗口: {len(X_te):,}")

    epoch_bar = tqdm(range(1, n_epochs + 1),
                     desc=f"{tag} 总进度", ncols=80, position=0)

    for epoch in epoch_bar:
        t0 = time.time()
        model.train()
        losses = []

        batch_bar = tqdm(tr_loader,
                         desc=f"  {tag} Epoch {epoch:02d}/{n_epochs} 训练",
                         ncols=80, leave=False, position=1)
        for xb, yb in batch_bar:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred, recon, _, cl = model(xb)
            loss = criterion(pred, yb, recon=recon, x=xb, contrast_loss=cl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = np.mean(losses)

        # ── 评估 AUC ──────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            tr_scores = collect_scores(tr_loader, desc=f"{tag} 训练集推理")
            te_scores = collect_scores(te_loader, desc=f"{tag} 测试集推理")

        # IQR 归一化（GDN 风格）
        tr_med = np.median(tr_scores, axis=0, keepdims=True)
        tr_iqr = (np.percentile(tr_scores, 75, axis=0, keepdims=True)
                  - np.percentile(tr_scores, 25, axis=0, keepdims=True) + 0.01)
        z_te = np.abs(te_scores - tr_med) / tr_iqr

        # top-3 mean 聚合 → 全局分数
        k = min(3, z_te.shape[1])
        global_score = np.partition(z_te, -k, axis=1)[:, -k:].mean(axis=1)

        # 对齐标签长度
        n = min(len(global_score), len(te_labels_aligned))
        labels_cut = te_labels_aligned[:n]
        score_cut  = global_score[:n]

        try:
            auc = roc_auc_score(labels_cut, score_cut)
        except Exception:
            auc = 0.5

        elapsed = time.time() - t0
        history["epoch"].append(epoch)
        history["auc"].append(auc)
        history["train_loss"].append(avg_loss)

        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", auc=f"{auc:.4f}")
        if verbose:
            tqdm.write(f"{tag} Epoch {epoch:02d}/{n_epochs}  "
                       f"loss={avg_loss:.4f}  AUC={auc:.4f}  ({elapsed:.1f}s)")

    epoch_bar.close()
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=8,    help="训练轮数（默认 8）")
    parser.add_argument("--n_channels", type=int, default=25,   help="通道数（默认 25，与 SMAP 一致）")
    parser.add_argument("--train_len",  type=int, default=5000, help="训练集长度")
    parser.add_argument("--test_len",   type=int, default=8000, help="测试集长度")
    args = parser.parse_args()

    print("=" * 60)
    print("  MambGAT-AD 快速验证：v1 vs v2")
    print(f"  epochs={args.epochs}  channels={args.n_channels}")
    print(f"  train_len={args.train_len}  test_len={args.test_len}")
    print("=" * 60)

    # 生成合成数据
    print("\n[Data] 生成合成数据...")
    train_data, test_data, test_labels = make_synthetic_data(
        n_channels=args.n_channels,
        train_len=args.train_len,
        test_len=args.test_len,
    )
    print(f"[Data] train={train_data.shape}  test={test_data.shape}  "
          f"anomaly_ratio={test_labels.mean():.3f}")

    # 运行 v1
    print("\n" + "─" * 60)
    print("  运行 v1（原始模型：单点投影，无频域/对比损失）")
    print("─" * 60)
    hist_v1 = run_experiment(
        "v1", args.n_channels, train_data, test_data, test_labels,
        n_epochs=args.epochs,
    )

    # 运行 v2
    print("\n" + "─" * 60)
    print("  运行 v2（改进模型：多尺度Patch + 频域损失 + 图对比）")
    print("─" * 60)
    hist_v2 = run_experiment(
        "v2", args.n_channels, train_data, test_data, test_labels,
        n_epochs=args.epochs,
    )

    # ── 汇总对比 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  结果汇总")
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

    if best_v2 > best_v1:
        print(f"\n  ✅ v2 改进有效！最佳 AUC 提升 {best_v2-best_v1:+.4f}")
        print("     建议：在服务器上用 config/smap.yaml 全量训练。")
    elif best_v2 > best_v1 - 0.02:
        print(f"\n  ⚠️  v2 与 v1 接近（差距 < 0.02），合成数据可能不足以区分。")
        print("     建议：直接在服务器上用真实 SMAP 数据全量训练验证。")
    else:
        print(f"\n  ❌ v2 在合成数据上未超过 v1，需要检查超参。")

    print("=" * 60)


if __name__ == "__main__":
    main()
