"""
评估指标

包含：
  • F1 / Precision / Recall
  • Point-Adjust F1（PA@k，业界常用）
  • 原始 F1（无 PA，更严格，推荐同时汇报）
  • AUC-ROC

关于 Point-Adjust（PA）的说明：
  如果预测结果在真实异常区间内至少有一个点被检测到，
  则该区间内所有点都算作被正确检测。
  这对于实际运维场景合理，但可能虚高分数。
  论文中建议同时汇报 PA 和非 PA 的 F1，表明研究的严谨性。
  参考：Kim et al., "Towards a Rigorous Evaluation of Time-Series
        Anomaly Detection", AAAI 2022

关于 PA%k：
  更公平的版本：要求异常区间内 k% 的点被检测到才算整段命中。
  推荐使用 k=0 (raw), k=100 (strict PA) 两个极端值同时汇报。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Point-Adjust（宽松评估）
# ─────────────────────────────────────────────────────────────────────────────

def point_adjust(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Point-Adjust：若预测在某异常区间内至少命中一个点，则整段都算命中。

    Args:
        y_true: (T,) 真实标签 0/1
        y_pred: (T,) 预测标签 0/1
    Returns:
        y_pred_adjusted: (T,) 调整后的预测标签
    """
    y_pred_adj = y_pred.copy()
    anomaly_state = False
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1 and not anomaly_state:
            # 命中了异常区间的开头，回溯填充整段
            anomaly_state = True
            for j in range(i, -1, -1):
                if y_true[j] == 0:
                    break
                y_pred_adj[j] = 1
        elif y_true[i] == 0:
            anomaly_state = False
        if anomaly_state:
            y_pred_adj[i] = 1
    return y_pred_adj


def point_adjust_pa_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: float = 0.5,
) -> np.ndarray:
    """
    PA%k：要求异常区间内至少 k 比例的点被检测到才整段调整。

    k=0.0 → 等同于标准 PA
    k=1.0 → 要求全段命中（最严格）
    """
    # 找出所有异常区间
    segments = _find_segments(y_true)
    # 先用最优阈值把 score 转 label
    threshold = np.percentile(y_score, 95)
    y_pred = (y_score > threshold).astype(int)
    y_pred_adj = y_pred.copy()

    for start, end in segments:
        seg_len = end - start
        hit_cnt = y_pred[start:end].sum()
        if seg_len > 0 and hit_cnt / seg_len >= k:
            y_pred_adj[start:end] = 1

    return y_pred_adj


def _find_segments(y_true: np.ndarray):
    """返回所有连续异常段的 (start, end) 列表"""
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(y_true):
        if v == 1 and not in_seg:
            start = i
            in_seg = True
        elif v == 0 and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(y_true)))
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# 主评估函数
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_anomaly(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,
    use_pa: bool = True,
) -> Dict[str, float]:
    """
    计算全套异常检测评估指标。

    Args:
        y_true:  (T,) 真实标签 0/1
        y_pred:  (T,) 预测标签 0/1（由阈值模块输出）
        y_score: (T,) 原始异常分数（用于 AUC 计算，可选）
        use_pa:  是否同时计算 Point-Adjust 版本

    Returns:
        指标字典，包含：
          f1_raw, prec_raw, rec_raw     — 原始（无PA）
          f1_pa, prec_pa, rec_pa        — Point-Adjust 版本
          auc_roc, auc_pr              — 分数曲线（需要 y_score）
    """
    results = {}

    # ── 原始（严格）指标 ─────────────────────────────────────────
    results["f1_raw"]   = f1_score(y_true, y_pred, zero_division=0)
    results["prec_raw"] = precision_score(y_true, y_pred, zero_division=0)
    results["rec_raw"]  = recall_score(y_true, y_pred, zero_division=0)

    # ── Point-Adjust 指标 ────────────────────────────────────────
    if use_pa:
        y_pred_pa = point_adjust(y_true, y_pred)
        results["f1_pa"]   = f1_score(y_true, y_pred_pa, zero_division=0)
        results["prec_pa"] = precision_score(y_true, y_pred_pa, zero_division=0)
        results["rec_pa"]  = recall_score(y_true, y_pred_pa, zero_division=0)

    # ── AUC（需要连续分数）──────────────────────────────────────
    if y_score is not None and y_true.sum() > 0:
        try:
            results["auc_roc"] = roc_auc_score(y_true, y_score)
            results["auc_pr"]  = average_precision_score(y_true, y_score)
        except Exception:
            results["auc_roc"] = 0.0
            results["auc_pr"]  = 0.0

    return results


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """格式化打印评估指标"""
    print(f"\n{'─'*55}")
    if prefix:
        print(f"  {prefix}")
    print(f"  {'指标':<18} {'值':>10}")
    print(f"{'─'*55}")
    order = [
        ("f1_pa",   "F1 (Point-Adjust)"),
        ("prec_pa", "Precision (PA)"),
        ("rec_pa",  "Recall (PA)"),
        ("f1_raw",  "F1 (Raw, 严格)"),
        ("prec_raw","Precision (Raw)"),
        ("rec_raw", "Recall (Raw)"),
        ("auc_roc", "AUC-ROC"),
        ("auc_pr",  "AUC-PR"),
    ]
    for key, label in order:
        if key in metrics:
            print(f"  {label:<18} {metrics[key]:>10.4f}")
    print(f"{'─'*55}")
