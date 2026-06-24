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
from utils.threshold import PerChannelThreshold


def evaluate(args):
    # ── 加载 checkpoint ───────────────────────────────────────────
    ckpt   = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg    = ckpt["cfg"]
    channels = ckpt.get("channels", [])

    device_str = cfg["train"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    # ── 数据 ──────────────────────────────────────────────────────
    train_loader, test_loader, test_labels, n_channels = build_loaders(
        data_dir    = cfg["data"]["data_dir"],
        label_file  = cfg["data"]["label_file"],
        dataset     = cfg["data"]["dataset"],
        window_size = cfg["data"]["window_size"],
        train_step  = cfg["data"].get("window_step", 5),
        test_step   = cfg["data"].get("test_step", 1),
        batch_size  = cfg["train"]["batch_size"],
        num_workers = cfg["train"].get("num_workers", 0),
    )
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
        dropout     = 0.0,  # 评估时关闭 dropout
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Eval] 加载 checkpoint: {args.ckpt}  (epoch {ckpt.get('epoch','?')})")

    # ── 收集误差 ──────────────────────────────────────────────────
    def collect(loader):
        all_s = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device, dtype=torch.float32)
                _, s = model(x)
                all_s.append(s.cpu().numpy())
        return np.concatenate(all_s, axis=0)

    print("[Eval] 推理中 ...")
    train_errors = collect(train_loader)
    test_errors  = collect(test_loader)

    # ── 评估（与 train.py 完全一致）──────────────────────────────
    import json as _json
    data_fmt     = cfg["data"].get("format", "AT").upper()
    percentile   = cfg.get("threshold", {}).get("percentile", 99.5)
    dataset_name = cfg["data"]["dataset"]
    test_len     = len(test_errors)
    out_dir      = Path(args.ckpt).parent

    if data_fmt == "AT":
        global_label = test_labels[:test_len].astype(int)
        global_score = test_errors.mean(axis=1)
        thr          = float(np.percentile(train_errors.mean(axis=1), percentile))
        global_pred  = (global_score > thr).astype(int)

        metrics = evaluate_anomaly(
            y_true=global_label, y_pred=global_pred,
            y_score=global_score, use_pa=True,
            dataset=dataset_name,
        )
        print_metrics(metrics, prefix=f"MambGAT-AD [{dataset_name.upper()}] [全局, AT格式]")
        all_results = {k: round(float(v), 6) for k, v in metrics.items()}
    else:
        from utils.metrics import evaluate_per_channel
        per_ch_labels = test_labels[:test_len]
        metrics = evaluate_per_channel(
            per_ch_labels, test_errors, train_errors, percentile)
        print_metrics(metrics, prefix=f"MambGAT-AD [{dataset_name.upper()}] [逐通道]")
        all_results = {k: round(float(v), 6) for k, v in metrics.items()
                       if isinstance(v, (int, float))}

    out_path = out_dir / f"eval_{dataset_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(all_results, f, indent=2)
    print(f"[Eval] 结果保存至 {out_path}")

    # ── 可视化传感器耦合图 ────────────────────────────────────────
    if args.plot_graph:
        _plot_adjacency(model, n_channels, out_dir, cfg["data"]["dataset"])

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
