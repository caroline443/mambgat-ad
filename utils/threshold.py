"""
动态阈值模块

包含三种阈值策略：
  1. TelemanomThreshold  — 原始 Telemanom 非参数自适应阈值（修复版）
  2. PercentileThreshold — 固定分位数阈值（简单基线）
  3. ValSetThreshold     — 验证集自适应阈值（推荐，AUC 最优）
     在验证集分数上搜索使 F1-PA 最大的阈值，避免 percentile 固定值
     与真实异常率不匹配的问题。

参考：
  Hundman et al., KDD 2018 (Telemanom)
  Deng & Hooi, AAAI 2021 (GDN)
  Kim et al., AAAI 2022 (PA 批评)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _count_sequences(arr: np.ndarray) -> int:
    count, in_seq = 0, False
    for v in arr:
        if v == 1 and not in_seq:
            count += 1; in_seq = True
        elif v == 0:
            in_seq = False
    return count


def _point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Point Adjustment：异常段内任意一点命中则整段算对。"""
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
# 1. Telemanom 阈值（修复版）
# ─────────────────────────────────────────────────────────────────────────────

class TelemanomThreshold:
    """
    Telemanom 非参数自适应阈值（修复 n_seqs^2 → n_seqs）。
    """

    def __init__(self, p: float = 0.13, error_buffer: int = 100):
        self.p = p
        self.error_buffer = error_buffer
        self.threshold_ = None

    def fit(self, errors: np.ndarray) -> "TelemanomThreshold":
        self.threshold_ = self._find_threshold(errors)
        return self

    def _smooth(self, errors: np.ndarray) -> np.ndarray:
        if self.error_buffer <= 1 or len(errors) < self.error_buffer:
            return errors
        kernel = np.ones(self.error_buffer) / self.error_buffer
        return np.convolve(errors, kernel, mode='same')

    def _find_threshold(self, errors: np.ndarray) -> float:
        smoothed = self._smooth(errors)
        sorted_e = np.sort(smoothed)
        max_score = -np.inf
        best_eps  = sorted_e[-1]

        start_idx = int(len(sorted_e) * (1 - self.p))
        candidates = sorted_e[start_idx:]

        for eps in candidates:
            is_anom = (smoothed > eps).astype(int)
            n_seqs = _count_sequences(is_anom)
            if n_seqs == 0:
                continue
            above = smoothed[smoothed > eps]
            reduction = (above - eps).sum() / (smoothed.sum() + 1e-8)
            score = reduction / max(1, n_seqs)
            if score > max_score:
                max_score = score
                best_eps  = eps

        return float(best_eps)

    def predict(self, errors: np.ndarray) -> np.ndarray:
        assert self.threshold_ is not None
        smoothed = self._smooth(errors)
        return (smoothed > self.threshold_).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 分位数阈值（简单基线）
# ─────────────────────────────────────────────────────────────────────────────

class PercentileThreshold:
    """取训练集误差第 percentile 分位数为阈值。简单、可解释、稳定。"""

    def __init__(self, percentile: float = 99.5):
        self.percentile = percentile
        self.threshold_ = None

    def fit(self, errors: np.ndarray) -> "PercentileThreshold":
        self.threshold_ = float(np.percentile(errors, self.percentile))
        return self

    def predict(self, errors: np.ndarray) -> np.ndarray:
        assert self.threshold_ is not None
        return (errors > self.threshold_).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 验证集自适应阈值（推荐）
# ─────────────────────────────────────────────────────────────────────────────

class ValSetThreshold:
    """
    在验证集分数上搜索使 F1-PA 最大的阈值。

    核心思路：
      - 用训练集后 val_ratio 比例的数据作为"伪验证集"
      - 在验证集分数上枚举候选阈值，选 F1-PA 最大的
      - 若验证集无标注（无监督场景），退化为异常率阈值：
        取 score 最高的 anomaly_ratio% 为异常

    Args:
        val_ratio:      从训练集末尾切出的验证比例（默认 0.2）
        anomaly_ratio:  数据集已知异常率，用于无标注退化模式
                        （SMAP=0.1313, MSL=0.1072, SMD=0.0416）
        n_candidates:   阈值搜索候选数（越多越精确，越慢）
        smooth_window:  对分数做滑动平均的窗口（0=不平滑）
    """

    # 各数据集标注异常率（来自 ContrastAD Table 1）
    ANOMALY_RATIO = {
        "smap": 0.1313,
        "msl":  0.1072,
        "smd":  0.0416,
        "psm":  0.2776,
        "swat": 0.1214,
    }

    def __init__(
        self,
        val_ratio: float = 0.2,
        anomaly_ratio: float = None,
        dataset: str = None,
        n_candidates: int = 300,
        smooth_window: int = 10,
    ):
        self.val_ratio          = val_ratio
        self.n_candidates       = n_candidates
        self.smooth_window      = smooth_window
        self.threshold_         = None
        self._anomaly_ratio_mode = False

        # 确定异常率
        if anomaly_ratio is not None:
            self.anomaly_ratio = anomaly_ratio
        elif dataset is not None:
            self.anomaly_ratio = self.ANOMALY_RATIO.get(
                dataset.lower(), 0.10
            )
        else:
            self.anomaly_ratio = 0.10   # 保守默认值

    def _smooth(self, scores: np.ndarray) -> np.ndarray:
        if self.smooth_window <= 1:
            return scores
        kernel = np.ones(self.smooth_window) / self.smooth_window
        return np.convolve(scores, kernel, mode='same')

    def fit(
        self,
        train_scores: np.ndarray,
        val_labels: Optional[np.ndarray] = None,
    ) -> "ValSetThreshold":
        """
        Args:
            train_scores: (T_train,) 训练集全局异常分数
            val_labels:   (T_val,)  验证集标注（可选）
                          若提供，在验证集上搜索最优 F1-PA 阈值；
                          若不提供，用异常率阈值（anomaly-ratio 协议）。
        """
        scores = self._smooth(train_scores)

        if val_labels is not None and val_labels.sum() > 0:
            # ── 有标注：搜索最优 F1-PA 阈值 ──────────────────────
            # 取训练集末尾 val_ratio 的分数作为验证集分数
            n_val = max(1, int(len(scores) * self.val_ratio))
            val_scores = scores[-n_val:]
            val_labels_cut = val_labels[-n_val:]

            candidates = np.unique(
                np.percentile(val_scores,
                              np.linspace(50, 100, self.n_candidates))
            )
            best_thr, best_f1 = candidates[-1], -1.0
            for thr in candidates:
                pred = (val_scores > thr).astype(int)
                pred_pa = _point_adjust(val_labels_cut, pred)
                tp = (pred_pa * val_labels_cut).sum()
                fp = (pred_pa * (1 - val_labels_cut)).sum()
                fn = ((1 - pred_pa) * val_labels_cut).sum()
                p  = tp / (tp + fp + 1e-8)
                r  = tp / (tp + fn + 1e-8)
                f1 = 2 * p * r / (p + r + 1e-8)
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr

            self.threshold_          = float(best_thr)
            self._anomaly_ratio_mode = False
            self._fit_mode           = f"val-F1-PA (best_f1={best_f1:.4f})"

        else:
            # ── 无标注：anomaly-ratio 模式 ────────────────────────────
            # threshold_ 占位（predict 时会从测试分数重新计算，见 predict()）
            self.threshold_          = 0.0   # 占位，不用于实际判断
            self._anomaly_ratio_mode = True
            self._fit_mode           = f"anomaly-ratio ({self.anomaly_ratio:.4f}, 从测试分数计算)"

        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Bug 3 修复：anomaly-ratio 模式下阈值从 **测试分数** 实时计算，
        与 ContrastAD / Anomaly Transformer 的标准协议完全一致。
        val-F1-PA 模式保持不变（使用 fit 期间搜到的最优阈值）。
        """
        assert self.threshold_ is not None, "请先调用 fit()"
        smoothed = self._smooth(scores)
        if getattr(self, "_anomaly_ratio_mode", False):
            # anomaly-ratio 协议：取测试分数 top anomaly_ratio% 为异常
            # 每次从输入分数现算，保证与论文协议一致
            pct = 100.0 * (1.0 - self.anomaly_ratio)
            thr = float(np.percentile(smoothed, pct))
            return (smoothed > thr).astype(int)
        return (smoothed > self.threshold_).astype(int)

    def __repr__(self):
        mode = getattr(self, "_fit_mode", "未拟合")
        return (f"ValSetThreshold(thr={self.threshold_:.6f}, "
                f"mode={mode})")


# ─────────────────────────────────────────────────────────────────────────────
# 逐通道阈值（外部调用入口）
# ─────────────────────────────────────────────────────────────────────────────

class PerChannelThreshold:
    """
    对每个通道独立拟合阈值。
    全局判断：任意通道报警 = 全局报警（用于全局 F1 参考）。
    """

    def __init__(self, method: str = "percentile", **kwargs):
        assert method in ("telemanom", "percentile")
        self.method = method
        self.telemanom_kwargs = {k: v for k, v in kwargs.items()
                                 if k in ("p", "error_buffer")}
        self.percentile_kwargs = {k: v for k, v in kwargs.items()
                                  if k in ("percentile",)}
        self.thresholds = []

    def _make(self):
        if self.method == "telemanom":
            return TelemanomThreshold(**self.telemanom_kwargs)
        return PercentileThreshold(**self.percentile_kwargs)

    def fit(self, train_errors: np.ndarray) -> "PerChannelThreshold":
        """train_errors: (T_train, N)"""
        self.thresholds = [self._make().fit(train_errors[:, i])
                           for i in range(train_errors.shape[1])]
        return self

    def predict(self, test_errors: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            per_channel: (T_test, N)
            global_pred: (T_test,)  任意通道报警 → 全局
        """
        per_ch = np.stack(
            [t.predict(test_errors[:, i])
             for i, t in enumerate(self.thresholds)],
            axis=1
        )
        return per_ch, per_ch.any(axis=1).astype(int)
