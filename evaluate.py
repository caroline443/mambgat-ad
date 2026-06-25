"""
MambGAT-AD 独立评估脚本

用于加载已训练的 checkpoint，在测试集上生成完整评估报告，
并可视化学习到的传感器耦合图（论文图表）。

用法：
  python evaluate.py --ckpt checkpoints/best_smap.pt
  python evaluate.py --ckpt checkpoints/best_smap.pt --plot_graph
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import build_loaders
from models import MambGATAD
from utils import evaluate_anomaly, print_metrics
from utils.threshold import ValSetThreshold


def topk_mean_agg(z: np.ndarray, k: int = 3) -> np.ndarray:
    """取 top-k 通道分数均值（与 train.py 保持完全一致）"""
    k = min(k, z.shape[1])
    topk = np.partition(z, -k, axis=1)[:, -k:]
    return topk.mean(axis=1)


def evaluate(args):
    # ── 加载 checkpoint ───────────────────────────────────────────
    ckpt     = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg      = ckpt["cfg"]
    channels = ckpt.get("channels", [])

    device_str = cfg["train"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    data_fmt    = cfg["data"].get("format", "AT").upper()
    window_size = cfg["data"]["window_size"]

    # ── Bug 2 修复：补全 fmt 和 normalize_data 参数 ───────────────
    # Bug 4 修复：train_step 默认值与 train.py 统一为 1
    train_loader, test_loader, test_labels, n_channels = build_loaders(
        data_dir       = cfg["data"]["data_dir"],
        dataset        = cfg["data"]["dataset"],
        fmt            = data_fmt,                          # ← 修复：显式传 fmt
        label_file     = cfg["data"].get("label_file"),
        window_size    = window_size,
        train_step     = cfg["data"].get("window_step", 1), # ← 修复：默认 1
        test_step      = cfg["data"].get("test_step", 1),
        batch_size     = cfg["train"]["batch_size"],
        normalize_data = cfg["data"].get("normalize", True), # ← 修复：显式传归一化
        num_workers    = 0,   # 评估时强制 0，避免多进程死锁
    )

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
        dropout     = 0.0,  # 评估时关闭 dropout
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Eval] 加载 checkpoint: {args.ckpt}  (epoch {ckpt.get('epoch','?')})")

    # ── 收集误差（带进度条）──────────────────────────────────────
    from tqdm import tqdm

    def collect(loader, desc="推理"):
        all_s = []
        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"[Eval] {desc}",
                              ncols=80, unit="batch"):
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                pred, recon, _, ___ = model(x)
                pred_err  = (pred.squeeze(-1) - y).abs()
                recon_err = (recon - x).abs().mean(dim=1)
                all_s.append((pred_err + recon_err).cpu().numpy())
        return np.concatenate(all_s, axis=0)

    train_errors = collect(train_loader, "训练集")
    test_errors  = collect(test_loader,  "测试集")

    # ── 评估（与 train.py 完全一致）──────────────────────────────
    thr_cfg      = cfg.get("threshold", {})
    percentile   = thr_cfg.get("percentile", 99.5)
    top_k        = thr_cfg.get("top_k", 3)
    dataset_name = cfg["data"]["dataset"]
    test_len     = len(test_errors)
    out_dir      = Path(args.ckpt).parent

    if data_fmt == "AT":
        # ── Bug 1 修复：标签对齐 ──────────────────────────────────
        # score[i] 来自窗口 data[i : i+W]，应对应标签 labels[i + W - 1]
        label_offset = window_size - 1
        label_end    = min(label_offset + test_len, len(test_labels))
        global_label = test_labels[label_offset:label_end].astype(int)
        if len(global_label) < test_len:
            test_len    = len(global_label)
            test_errors = test_errors[:test_len]

        # GDN 风格 IQR 归一化
        tr_median = np.median(train_errors, axis=0, keepdims=True)
        tr_iqr    = (np.percentile(train_errors, 75, axis=0, keepdims=True)
                     - np.percentile(train_errors, 25, axis=0, keepdims=True) + 0.01)
        z_test    = np.abs(test_errors  - tr_median) / tr_iqr
        z_train   = np.abs(train_errors - tr_median) / tr_iqr

        # Bug 5 修复：top-k mean 聚合
        global_score   = topk_mean_agg(z_test,  k=top_k)
        train_score_1d = topk_mean_agg(z_train, k=top_k)

        # Bug 3 修复：anomaly-ratio 阈值从 test scores 确定（ContrastAD 协议）
        val_thr = ValSetThreshold(
            dataset       = dataset_name,
            smooth_window = thr_cfg.get("smooth_window", 10),
            n_candidates  = thr_cfg.get("n_candidates", 300),
        ).fit(train_score_1d)
        global_pred = val_thr.predict(global_score)
        print(f"[Threshold] {val_thr}")

        metrics = evaluate_anomaly(
            y_true=global_label, y_pred=global_pred,
            y_score=global_score, use_pa=True,
            dataset=dataset_name,
        )
        print_metrics(metrics, prefix=f"MambGAT-AD [{dataset_name.upper()}] [全局, AT格式]")
        all_results = {k: round(float(v), 6) for k, v in metrics.items()}

    else:
        # Telemanom 格式：逐通道宏平均
        from utils.metrics import evaluate_per_channel
        per_ch_labels = test_labels[:test_len]
        metrics = evaluate_per_channel(
            per_ch_labels, test_errors, train_errors, percentile)
        print_metrics(metrics, prefix=f"MambGAT-AD [{dataset_name.upper()}] [逐通道]")
        all_results = {k: round(float(v), 6) for k, v in metrics.items()
                       if isinstance(v, (int, float))}

    out_path = out_dir / f"eval_{dataset_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Eval] 结果保存至 {out_path}")

    if args.plot_graph:
        _plot_adjacency(model, n_channels, out_dir, dataset_name)

    return metrics


def _plot_adjacency(model, n_channels, out_dir, dataset_name):
    """绘制学习到的传感器耦合图（论文 Figure 用）"""
    try:
        import matplotlib.pyplot as plt

        adj = model.get_graph(head_idx=0).numpy()  # (N, N)
        N   = n_channels

        fig, ax = plt.subplots(figsize=(max(8, N * 0.18), max(8, N * 0.18)))
        im = ax.imshow(adj, cmap="viridis", aspect="auto")
        ax.set_title(f"MambGAT-AD 学习到的传感器耦合图\n({dataset_name.upper()}, {N} 通道)", pad=10)
        ax.set_xlabel("目标节点")
        ax.set_ylabel("源节点")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        fig_path = out_dir / f"graph_{dataset_name}.pdf"
        plt.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[Eval] 耦合图已保存 → {fig_path}")
    except Exception as e:
        print(f"[WARN] 绘图失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MambGAT-AD 评估脚本")
    parser.add_argument("--ckpt",       required=True, help="checkpoint 路径")
    parser.add_argument("--plot_graph", action="store_true", help="输出传感器耦合图")
    args = parser.parse_args()
    evaluate(args)
