"""
动态阈值模块（修复版）

修复内容：
  - Telemanom：n_seqs^2 改为 n_seqs（线性惩罚，不过度压低阈值）
  - 新增 PercentileThreshold（默认推荐，简单可靠）
  - 评估改用逐通道模式，不再依赖全局阈值

参考：Hundman et al., KDD 2018
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Telemanom 阈值（修复版）
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
            # 修复：线性惩罚而不是平方，避免过度压低阈值
            score = reduction / max(1, n_seqs)
            if score > max_score:
                max_score = score
                best_eps  = eps

        return float(best_eps)

    def predict(self, errors: np.ndarray) -> np.ndarray:
        assert self.threshold_ is not None
        smoothed = self._smooth(errors)
        return (smoothed > self.threshold_).astype(int)


def _count_sequences(arr: np.ndarray) -> int:
    count, in_seq = 0, False
    for v in arr:
        if v == 1 and not in_seq:
            count += 1; in_seq = True
        elif v == 0:
            in_seq = False
    return count


# ─────────────────────────────────────────────────────────────────────────────
# 分位数阈值（推荐默认）
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
