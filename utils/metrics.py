"""
评估指标

包含：
  • F1 / Precision / Recall（raw，无 PA）
  • Point-Adjust F1（PA，业界常用宽松版）
  • VUS-ROC / VUS-PR（VLDB 2022，最新严格版，SOTA 论文使用）
  • AUC-ROC / AUC-PR

指标选型建议（投顶会）：
  - F1(PA)   对标老 baseline（Telemanom、GDN、OmniAnomaly）
  - F1(raw)  证明严谨性，防评审质疑 PA 虚高
  - VUS-ROC  对标 2026 SOTA（Multi-View Channel-Graph 报 0.675）

参考文献：
  Kim et al., "Towards a Rigorous Evaluation of Time-Series Anomaly
              Detection", AAAI 2022  （批评 PA 虚高）
  Paparrizos et al., "Volume Under the Surface: A New Accuracy
              Evaluation Measure for TSAD", VLDB 2022  （提出 VUS）
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Point-Adjust（宽松评估）
# ─────────────────────────────────────────────────────────────────────────────

def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    若预测在某异常区间内至少命中一个点，则整段都算命中。
    """
    y_pred_adj = y_pred.copy()
    anomaly_state = False
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1 and not anomaly_state:
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


# ─────────────────────────────────────────────────────────────────────────────
# VUS-ROC / VUS-PR（VLDB 2022）
# ─────────────────────────────────────────────────────────────────────────────

def _buffer_labels(y_true: np.ndarray, delta: int) -> np.ndarray:
    """
    将标签向两侧各扩展 delta 个时间步（容忍窗口）。
    anomaly_score 在 delta 范围内命中也算 TP。
    """
    if delta == 0:
        return y_true.copy()
    buffered = y_true.copy()
    anom_idx = np.where(y_true == 1)[0]
    for idx in anom_idx:
        lo = max(0, idx - delta)
        hi = min(len(y_true), idx + delta + 1)
        buffered[lo:hi] = 1
    return buffered


def vus_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_buffer: int = 100,
) -> float:
    """
    VUS-ROC：在不同容忍窗口 δ ∈ [0, max_buffer] 下计算 AUC-ROC，取平均。

    δ=0：严格逐点 AUC（等同于标准 AUC-ROC）
    δ 越大：越宽松

    参考：Paparrizos et al., VLDB 2022
    """
    if y_true.sum() == 0:
        return 0.0

    aucs = []
    # 采样 21 个 δ 值（0, 5, 10, ..., max_buffer），平衡精度与速度
    deltas = np.linspace(0, max_buffer, num=21, dtype=int)
    for delta in deltas:
        buf = _buffer_labels(y_true, int(delta))
        if buf.sum() == 0 or buf.sum() == len(buf):
            continue
        try:
            aucs.append(roc_auc_score(buf, y_score))
        except Exception:
            pass

    return float(np.mean(aucs)) if aucs else 0.0


def vus_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_buffer: int = 100,
) -> float:
    """
    VUS-PR：在不同容忍窗口 δ 下计算 AUC-PR，取平均。
    """
    if y_true.sum() == 0:
        return 0.0

    aucs = []
    deltas = np.linspace(0, max_buffer, num=21, dtype=int)
    for delta in deltas:
        buf = _buffer_labels(y_true, int(delta))
        if buf.sum() == 0 or buf.sum() == len(buf):
            continue
        try:
            aucs.append(average_precision_score(buf, y_score))
        except Exception:
            pass

    return float(np.mean(aucs)) if aucs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 主评估函数
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_anomaly(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,
    use_pa: bool = True,
    vus_max_buffer: int = 100,
) -> Dict[str, float]:
    """
    完整异常检测评估，输出论文所需全套指标。

    Args:
        y_true:          (T,) 真实标签 0/1
        y_pred:          (T,) 预测标签 0/1
        y_score:         (T,) 连续异常分数（用于 AUC/VUS 计算）
        use_pa:          是否计算 Point-Adjust 版本
        vus_max_buffer:  VUS 最大容忍窗口大小

    Returns:
        指标字典
    """
    results = {}

    # ── 原始（严格）指标 ────────────────────────────────────────
    results["f1_raw"]   = f1_score(y_true, y_pred, zero_division=0)
    results["prec_raw"] = precision_score(y_true, y_pred, zero_division=0)
    results["rec_raw"]  = recall_score(y_true, y_pred, zero_division=0)

    # ── Point-Adjust 指标 ───────────────────────────────────────
    if use_pa:
        y_pred_pa = point_adjust(y_true, y_pred)
        results["f1_pa"]   = f1_score(y_true, y_pred_pa, zero_division=0)
        results["prec_pa"] = precision_score(y_true, y_pred_pa, zero_division=0)
        results["rec_pa"]  = recall_score(y_true, y_pred_pa, zero_division=0)

    # ── AUC + VUS（需要连续分数）────────────────────────────────
    if y_score is not None and y_true.sum() > 0:
        try:
            results["auc_roc"] = roc_auc_score(y_true, y_score)
            results["auc_pr"]  = average_precision_score(y_true, y_score)
        except Exception:
            results["auc_roc"] = 0.0
            results["auc_pr"]  = 0.0

        # VUS（计算较慢，约数秒）
        results["vus_roc"] = vus_roc(y_true, y_score, vus_max_buffer)
        results["vus_pr"]  = vus_pr(y_true, y_score, vus_max_buffer)

    return results


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """格式化打印评估指标（论文 Table 风格）"""
    print(f"\n{'─'*58}")
    if prefix:
        print(f"  {prefix}")
    print(f"  {'指标':<22} {'值':>10}")
    print(f"{'─'*58}")
    order = [
        ("vus_roc",  "VUS-ROC  ★ 最新SOTA对标"),
        ("vus_pr",   "VUS-PR"),
        ("f1_pa",    "F1  (Point-Adjust)"),
        ("prec_pa",  "Precision (PA)"),
        ("rec_pa",   "Recall    (PA)"),
        ("f1_raw",   "F1  (Raw, 严格)"),
        ("prec_raw", "Precision (Raw)"),
        ("rec_raw",  "Recall    (Raw)"),
        ("auc_roc",  "AUC-ROC"),
        ("auc_pr",   "AUC-PR"),
    ]
    for key, label in order:
        if key in metrics:
            print(f"  {label:<22} {metrics[key]:>10.4f}")
    print(f"{'─'*58}")
