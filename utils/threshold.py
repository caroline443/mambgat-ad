"""
动态阈值模块

两种方法：
  1. TelemanomThreshold — 复现 Telemanom 的非参数自适应阈值
     通过寻找误差序列中"最佳分割点"自动确定阈值，无需手工设置
     参考：Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs
           and Nonparametric Dynamic Thresholding", KDD 2018

  2. PercentileThreshold — 简单分位数阈值（快速验证用）
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. Telemanom 动态阈值
# ─────────────────────────────────────────────────────────────────────────────

class TelemanomThreshold:
    """
    Telemanom 非参数自适应阈值。

    核心思想：
      对误差序列 e = [e_1, e_2, ..., e_T]，排序后遍历所有候选阈值 ε，
      寻找使"误差降低程度 / 异常区间数量"最大的 ε 作为最终阈值。
      这等价于在"减少假阳性"和"捕捉异常"之间自动平衡。

    Args:
        p:            控制平滑度的参数（默认 0.13，原论文推荐值）
        error_buffer: 预测误差平滑窗口大小
    """

    def __init__(self, p: float = 0.13, error_buffer: int = 100):
        self.p = p
        self.error_buffer = error_buffer
        self.threshold_ = None

    def fit(self, errors: np.ndarray) -> "TelemanomThreshold":
        """
        Args:
            errors: (T,) 一维误差序列（训练集上的预测误差）
        """
        self.mean_e    = np.mean(errors)
        self.std_e     = np.std(errors)
        self.threshold_ = self._find_threshold(errors)
        return self

    def _smooth(self, errors: np.ndarray) -> np.ndarray:
        """简单移动平均平滑"""
        if self.error_buffer <= 1:
            return errors
        kernel = np.ones(self.error_buffer) / self.error_buffer
        return np.convolve(errors, kernel, mode='same')

    def _find_threshold(self, errors: np.ndarray) -> float:
        """遍历候选阈值，寻找最优分割"""
        smoothed = self._smooth(errors)
        sorted_e = np.sort(smoothed)
        max_score = -np.inf
        best_eps  = sorted_e[-1]

        # 候选阈值从高到低遍历（取 top 5% 分位数附近）
        candidates = sorted_e[int(len(sorted_e) * (1 - self.p)):]

        for eps in candidates:
            is_anomaly = (smoothed > eps).astype(int)
            # 异常区间数
            n_seqs = _count_sequences(is_anomaly)
            if n_seqs == 0:
                continue
            # 超过阈值的误差降低比例
            above = smoothed[smoothed > eps]
            reduction = (above - eps).sum() / (smoothed.sum() + 1e-8)
            score = reduction / (n_seqs ** 2)
            if score > max_score:
                max_score = score
                best_eps  = eps

        return float(best_eps)

    def predict(self, errors: np.ndarray) -> np.ndarray:
        """
        Args:
            errors: (T,) 测试误差序列
        Returns:
            labels: (T,) 0/1 逐点预测标签
        """
        assert self.threshold_ is not None, "请先调用 fit()"
        smoothed = self._smooth(errors)
        return (smoothed > self.threshold_).astype(int)


def _count_sequences(binary_arr: np.ndarray) -> int:
    """统计连续 1 的段数"""
    count = 0
    in_seq = False
    for v in binary_arr:
        if v == 1 and not in_seq:
            count += 1
            in_seq = True
        elif v == 0:
            in_seq = False
    return count


# ─────────────────────────────────────────────────────────────────────────────
# 2. 简单分位数阈值
# ─────────────────────────────────────────────────────────────────────────────

class PercentileThreshold:
    """
    分位数阈值：取训练误差的第 percentile 分位数作为阈值。

    快速验证用，不需要调参。
    """

    def __init__(self, percentile: float = 99.5):
        self.percentile  = percentile
        self.threshold_  = None

    def fit(self, errors: np.ndarray) -> "PercentileThreshold":
        self.threshold_ = float(np.percentile(errors, self.percentile))
        return self

    def predict(self, errors: np.ndarray) -> np.ndarray:
        assert self.threshold_ is not None
        return (errors > self.threshold_).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 多通道阈值（自动对每个通道独立拟合）
# ─────────────────────────────────────────────────────────────────────────────

class PerChannelThreshold:
    """
    对每个通道独立拟合阈值，然后取"任意通道报警即为全局报警"。
    适合 SMAP/MSL 这类多通道数据集。
    """

    def __init__(self, method: str = "telemanom", **kwargs):
        assert method in ("telemanom", "percentile")
        self.method = method
        self.kwargs = kwargs
        self.thresholds = []

    def _make(self):
        if self.method == "telemanom":
            return TelemanomThreshold(**self.kwargs)
        return PercentileThreshold(**self.kwargs)

    def fit(self, train_errors: np.ndarray) -> "PerChannelThreshold":
        """
        Args:
            train_errors: (T_train, N) 训练集各通道误差
        """
        N = train_errors.shape[1]
        self.thresholds = []
        for i in range(N):
            t = self._make()
            t.fit(train_errors[:, i])
            self.thresholds.append(t)
        return self

    def predict(self, test_errors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            per_channel: (T_test, N)  每通道的逐点标签
            global_pred: (T_test,)    任意通道报警即全局报警
        """
        per_ch = []
        for i, t in enumerate(self.thresholds):
            per_ch.append(t.predict(test_errors[:, i]))
        per_channel = np.stack(per_ch, axis=1)          # (T, N)
        global_pred = per_channel.any(axis=1).astype(int)  # (T,)
        return per_channel, global_pred
