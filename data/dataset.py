"""
数据加载模块 — 支持 Telemanom / Kaggle 格式

数据目录结构（Kaggle: patrickfleith/nasa-anomaly-detection-dataset-smap-msl）：
  datasets/archive/
    data/
      train/   A-1.npy  P-1.npy  M-1.npy  ...（SMAP + MSL 混在一起）
      test/    A-1.npy  P-1.npy  M-1.npy  ...
    labeled_anomalies.csv   （含 spacecraft 列，区分 SMAP / MSL）

每个 .npy 文件 shape: (timesteps, n_features)
  - 第 0 列：主遥测值（我们取这一列）
  - 其余列：命令序列上下文特征
"""

from __future__ import annotations

import ast
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# 通道发现：从 labeled_anomalies.csv 读取属于指定数据集的通道名
# ─────────────────────────────────────────────────────────────────────────────

def get_channels(label_file: str, dataset: str) -> List[str]:
    """
    从 labeled_anomalies.csv 的 spacecraft 列筛选通道。
    dataset: "smap" 或 "msl"（大小写不敏感）
    """
    df = pd.read_csv(label_file)
    mask = df["spacecraft"].str.upper() == dataset.upper()
    channels = df.loc[mask, "chan_id"].tolist()
    return channels


# ─────────────────────────────────────────────────────────────────────────────
# 加载单通道
# ─────────────────────────────────────────────────────────────────────────────

def _load_channel(data_dir: str, split: str, chan_id: str) -> np.ndarray:
    """
    加载一个通道的 .npy，取第 0 列（主遥测值），返回 shape (T,)
    """
    path = os.path.join(data_dir, split, f"{chan_id}.npy")
    arr = np.load(path)          # (T, n_features)
    return arr[:, 0].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 加载多通道并对齐长度
# ─────────────────────────────────────────────────────────────────────────────

def load_multivariate(
    data_dir: str,
    split: str,
    channels: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    加载所有通道并堆叠为多变量时间序列。

    Returns:
        data:     (T, N)  float32
        channels: 实际成功加载的通道名列表（跳过缺失文件）
    """
    series, valid = [], []
    for ch in channels:
        path = os.path.join(data_dir, split, f"{ch}.npy")
        if not os.path.exists(path):
            print(f"  [跳过] {ch}.npy 不存在")
            continue
        series.append(_load_channel(data_dir, split, ch))
        valid.append(ch)

    if not series:
        raise RuntimeError(f"在 {data_dir}/{split}/ 中没有找到任何通道文件！")

    min_len = min(s.shape[0] for s in series)
    data = np.stack([s[:min_len] for s in series], axis=1)  # (T, N)
    return data, valid


# ─────────────────────────────────────────────────────────────────────────────
# 构建测试集逐点标签
# ─────────────────────────────────────────────────────────────────────────────

def build_labels(
    label_file: str,
    channels: List[str],
    test_len: int,
) -> np.ndarray:
    """
    从 labeled_anomalies.csv 构建全局逐点二值标签。
    任意通道在时刻 t 有异常 → labels[t] = 1

    Returns:
        labels: (test_len,)  int32
    """
    df = pd.read_csv(label_file)
    labels = np.zeros(test_len, dtype=np.int32)

    for ch in channels:
        row = df[df["chan_id"] == ch]
        if row.empty:
            continue
        seqs = ast.literal_eval(row.iloc[0]["anomaly_sequences"])
        for start, end in seqs:
            s = min(int(start), test_len - 1)
            e = min(int(end),   test_len)
            labels[s:e] = 1

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 归一化
# ─────────────────────────────────────────────────────────────────────────────

def normalize(train: np.ndarray, test: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std  = train.std( axis=0, keepdims=True) + 1e-8
    return (train - mean) / std, (test - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    滑动窗口数据集。

    样本：
      x: (window_size, N)  历史窗口
      y: (N,)              下一时间步（预测目标）
    """

    def __init__(self, data: np.ndarray, window_size: int, step: int = 1):
        super().__init__()
        self.data    = torch.from_numpy(data)
        self.ws      = window_size
        self.indices = list(range(0, len(data) - window_size, step))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.indices[idx]
        return self.data[s : s + self.ws], self.data[s + self.ws]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(
    data_dir: str,
    label_file: str,
    dataset: str = "smap",
    window_size: int = 100,
    train_step: int = 5,
    test_step: int = 1,
    batch_size: int = 64,
    normalize_data: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, np.ndarray, int]:
    """
    Returns:
        train_loader, test_loader,
        test_labels (T_test,) int32,
        n_channels int
    """
    # 1. 从 CSV 获取该数据集的通道列表
    channels = get_channels(label_file, dataset)
    print(f"[Data] {dataset.upper()} | CSV 中共 {len(channels)} 个通道")

    # 2. 加载多变量序列
    train_raw, channels = load_multivariate(data_dir, "train", channels)
    test_raw,  _        = load_multivariate(data_dir, "test",  channels)
    n_channels = len(channels)

    # 3. 归一化
    if normalize_data:
        train_raw, test_raw = normalize(train_raw, test_raw)

    # 4. 标签
    test_labels = build_labels(label_file, channels, len(test_raw))
    anomaly_rate = test_labels.mean()

    print(f"[Data] 通道数={n_channels} | "
          f"训练={len(train_raw):,} | 测试={len(test_raw):,} | "
          f"异常率={anomaly_rate:.2%}")

    # 5. Dataset / DataLoader
    train_ds = TimeSeriesDataset(train_raw, window_size, step=train_step)
    test_ds  = TimeSeriesDataset(test_raw,  window_size, step=test_step)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(num_workers > 0),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0),
    )

    return train_loader, test_loader, test_labels, n_channels
