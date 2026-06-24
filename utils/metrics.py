"""
评估指标

标准评估协议（类 Telemanom）：
  - 逐通道独立评估，报告宏平均（macro average）
  - 同时报告 F1(raw)、F1(PA)、VUS-ROC
  - 不用 OR 合并后的全局标签（会虚高异常率）

参考：
  Hundman et al., KDD 2018 (Telemanom)
  Kim et al., AAAI 2022 (批评 PA 虚高)
  Paparrizos et al., VLDB 2022 (VUS)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Point-Adjust
# ─────────────────────────────────────────────────────────────────────────────

def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_adj = y_pred.copy()
    in_anomaly = False
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1 and not in_anomaly:
            in_anomaly = True
            for j in range(i, -1, -1):
                if y_true[j] == 0:
                    break
                y_adj[j] = 1
        elif y_true[i] == 0:
            in_anomaly = False
        if in_anomaly:
            y_adj[i] = 1
    return y_adj


# ─────────────────────────────────────────────────────────────────────────────
# VUS-ROC / VUS-PR
# ─────────────────────────────────────────────────────────────────────────────

def _buffer_labels(y_true: np.ndarray, delta: int) -> np.ndarray:
    if delta == 0:
        return y_true.copy()
    buf = y_true.copy()
    for idx in np.where(y_true == 1)[0]:
        buf[max(0, idx - delta):min(len(y_true), idx + delta + 1)] = 1
    return buf


def vus_roc(y_true: np.ndarray, y_score: np.ndarray,
            max_buffer: int = 100) -> float:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.0
    aucs = []
    for delta in np.linspace(0, max_buffer, num=21, dtype=int):
        buf = _buffer_labels(y_true, int(delta))
        if 0 < buf.sum() < len(buf):
            try:
                aucs.append(roc_auc_score(buf, y_score))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


def vus_pr(y_true: np.ndarray, y_score: np.ndarray,
           max_buffer: int = 100) -> float:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.0
    aucs = []
    for delta in np.linspace(0, max_buffer, num=21, dtype=int):
        buf = _buffer_labels(y_true, int(delta))
        if 0 < buf.sum() < len(buf):
            try:
                aucs.append(average_precision_score(buf, y_score))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 异常率阈值（Anomaly Transformer / ContrastAD 协议，可与论文直接比 F1）
# ─────────────────────────────────────────────────────────────────────────────

# 各数据集的标注异常率（来自 AT 格式数据集统计，ContrastAD Table 1）
ANOMALY_RATIO = {
    "smap": 0.1313,
    "msl":  0.1072,
    "smd":  0.0416,
    "psm":  0.2776,
    "swat": 0.1214,
}

def anomaly_ratio_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    dataset: str = None,
    ratio: float = None,
) -> np.ndarray:
    """
    按已知异常率设阈值（Anomaly Transformer / ContrastAD 标准协议）。
    取 score 最高的 ratio% 为异常，其余为正常。

    与 ContrastAD Table 2 的 F1 数字直接可比。

    Args:
        dataset: 数据集名称（自动查异常率），或手动传 ratio
        ratio:   异常比例，如 0.1313 表示 13.13%
    """
    if ratio is None:
        ratio = ANOMALY_RATIO.get(dataset.lower() if dataset else '', y_true.mean())
    thr   = np.percentile(y_score, 100 * (1 - ratio))
    return (y_score > thr).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 最优阈值搜索
# ─────────────────────────────────────────────────────────────────────────────

def best_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_candidates: int = 200,
) -> Tuple[float, float]:
    """
    在测试集分数上搜索使 F1-PA 最大的阈值（论文常用的"上帝视角"协议）。
    返回 (best_threshold, best_f1_pa)
    注意：这是论文对齐用的参考值，不是真实部署值。
    """
    candidates = np.unique(np.percentile(y_score, np.linspace(0, 100, n_candidates)))
    best_thr, best_f1 = candidates[-1], 0.0
    for thr in candidates:
        pred = (y_score > thr).astype(int)
        pred_pa = point_adjust(y_true, pred)
        f1 = f1_score(y_true, pred_pa, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


# ─────────────────────────────────────────────────────────────────────────────
# 单通道评估
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_channel(
    y_true: np.ndarray,      # (T,)
    y_score: np.ndarray,     # (T,) 连续分数
    train_score: np.ndarray, # (T_train,) 训练集分数（用于阈值搜索）
    percentile: float = 99.5,
) -> Dict[str, float]:
    """
    对单个通道进行评估。
    阈值：先用 percentile 方法，同时搜索最优 F1 阈值。
    """
    if y_true.sum() == 0:
        return {}   # 该通道无标注异常，跳过

    results = {}

    # ── percentile 阈值 ──────────────────────────────────────────
    thr_pct  = float(np.percentile(train_score, percentile))
    y_pred   = (y_score > thr_pct).astype(int)
    y_pred_pa = point_adjust(y_true, y_pred)

    results["f1_raw"]   = f1_score(y_true, y_pred,    zero_division=0)
    results["prec_raw"] = precision_score(y_true, y_pred,    zero_division=0)
    results["rec_raw"]  = recall_score(y_true, y_pred,    zero_division=0)
    results["f1_pa"]    = f1_score(y_true, y_pred_pa, zero_division=0)
    results["prec_pa"]  = precision_score(y_true, y_pred_pa, zero_division=0)
    results["rec_pa"]   = recall_score(y_true, y_pred_pa, zero_division=0)

    # ── VUS ─────────────────────────────────────────────────────
    results["vus_roc"] = vus_roc(y_true, y_score)
    results["vus_pr"]  = vus_pr(y_true, y_score)

    # ── AUC ─────────────────────────────────────────────────────
    try:
        results["auc_roc"] = roc_auc_score(y_true, y_score)
        results["auc_pr"]  = average_precision_score(y_true, y_score)
    except Exception:
        pass

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 宏平均评估（所有通道独立评估后取平均）
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_per_channel(
    per_channel_labels: np.ndarray,  # (T_test, N)
    test_errors: np.ndarray,          # (T_test, N)
    train_errors: np.ndarray,         # (T_train, N)
    percentile: float = 99.5,
) -> Dict[str, float]:
    """
    逐通道评估，宏平均。这是标准的 SMAP 评估协议。
    只统计有标注异常的通道（跳过全零标签的通道）。
    """
    n_channels = per_channel_labels.shape[1]
    all_metrics: List[Dict] = []

    for i in range(n_channels):
        m = evaluate_channel(
            per_channel_labels[:, i],
            test_errors[:, i],
            train_errors[:, i],
            percentile=percentile,
        )
        if m:
            all_metrics.append(m)

    if not all_metrics:
        return {}

    # 宏平均
    keys = set().union(*all_metrics)
    macro = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        if vals:
            macro[k] = float(np.mean(vals))

    macro["n_channels_evaluated"] = len(all_metrics)
    return macro


# ─────────────────────────────────────────────────────────────────────────────
# 单次全局评估（兼容旧接口）
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_anomaly(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,
    use_pa: bool = True,
    vus_max_buffer: int = 100,
    dataset: str = None,      # 传入数据集名可自动计算 anomaly-ratio F1
) -> Dict[str, float]:
    results = {}
    results["f1_raw"]   = f1_score(y_true, y_pred, zero_division=0)
    results["prec_raw"] = precision_score(y_true, y_pred, zero_division=0)
    results["rec_raw"]  = recall_score(y_true, y_pred, zero_division=0)
    if use_pa:
        y_pa = point_adjust(y_true, y_pred)
        results["f1_pa"]   = f1_score(y_true, y_pa, zero_division=0)
        results["prec_pa"] = precision_score(y_true, y_pa, zero_division=0)
        results["rec_pa"]  = recall_score(y_true, y_pa, zero_division=0)
    if y_score is not None and y_true.sum() > 0:
        try:
            results["auc_roc"] = roc_auc_score(y_true, y_score)
            results["auc_pr"]  = average_precision_score(y_true, y_score)
        except Exception:
            pass
        results["vus_roc"] = vus_roc(y_true, y_score, vus_max_buffer)
        results["vus_pr"]  = vus_pr(y_true, y_score, vus_max_buffer)
        # best-F1 搜索（部分论文使用）
        if use_pa:
            _, best_f1 = best_f1_threshold(y_true, y_score)
            results["f1_pa_best"] = best_f1
        # anomaly-ratio 阈值（Anomaly Transformer / ContrastAD 协议，可直接对标论文）
        if dataset is not None:
            y_pred_ar = anomaly_ratio_threshold(y_true, y_score, dataset=dataset)
            y_pred_ar_pa = point_adjust(y_true, y_pred_ar)
            results["f1_pa_ar"]   = f1_score(y_true, y_pred_ar_pa, zero_division=0)
            results["prec_pa_ar"] = precision_score(y_true, y_pred_ar_pa, zero_division=0)
            results["rec_pa_ar"]  = recall_score(y_true, y_pred_ar_pa, zero_division=0)
    return results


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    n = metrics.get("n_channels_evaluated", "")
    suffix = f"  (评估通道数: {int(n)})" if n else ""
    print(f"\n{'─'*60}")
    if prefix:
        print(f"  {prefix}{suffix}")
    print(f"  {'指标':<22} {'值':>10}")
    print(f"{'─'*60}")
    order = [
        ("auc_roc",    "AUC-ROC ★ 与ContrastAD直接比"),
        ("vus_roc",    "VUS-ROC ★ 与Multi-View直接比"),
        ("f1_pa_ar",   "F1-PA (anomaly-ratio) ★直接比"),
        ("prec_pa_ar", "Precision (anomaly-ratio)"),
        ("rec_pa_ar",  "Recall    (anomaly-ratio)"),
        ("vus_pr",     "VUS-PR"),
        ("f1_pa_best", "F1-PA (best-F1, 参考)"),
        ("f1_pa",      "F1-PA (train-pct, 部署)"),
        ("prec_pa",    "Precision (PA)"),
        ("rec_pa",     "Recall    (PA)"),
        ("f1_raw",     "F1  (Raw)"),
        ("auc_pr",     "AUC-PR"),
    ]
    for key, label in order:
        if key in metrics:
            print(f"  {label:<22} {metrics[key]:>10.4f}")
    print(f"{'─'*60}")


