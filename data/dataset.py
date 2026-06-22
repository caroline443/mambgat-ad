"""
数据加载模块 — 支持 Anomaly Transformer 格式（推荐）

数据来源（Google Drive，公开可下载）：
  https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm

下载后放置如下：
  datasets/
    SMAP_train.npy       shape: (train_len, n_channels)
    SMAP_test.npy        shape: (test_len,  n_channels)
    SMAP_test_label.npy  shape: (test_len,)   0=正常 1=异常
    MSL_train.npy
    MSL_test.npy
    MSL_test_label.npy

格式说明：
  - 每个 .npy 文件包含所有通道，无需按通道分文件
  - 标签文件是逐点的全局标签（任意通道异常=1）
  - 数据已预处理，无需额外归一化（但本模块可选做 z-score）
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    data_dir: str,
    dataset: str,           # "smap" | "msl"
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载训练集、测试集和测试标签。

    Returns:
        train:  (T_train, N)  float32
        test:   (T_test,  N)  float32
        labels: (T_test,)     int32,  0=正常 1=异常
    """
    name = dataset.upper()   # SMAP / MSL
    train_path  = os.path.join(data_dir, f"{name}_train.npy")
    test_path   = os.path.join(data_dir, f"{name}_test.npy")
    label_path  = os.path.join(data_dir, f"{name}_test_label.npy")

    for p in [train_path, test_path, label_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"找不到文件: {p}\n"
                f"请从 Google Drive 下载数据后放入 {data_dir}/\n"
                f"下载地址: https://drive.google.com/drive/folders/"
                f"1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm"
            )

    train  = np.load(train_path).astype(np.float32)
    test   = np.load(test_path ).astype(np.float32)
    labels = np.load(label_path).astype(np.int32)

    if normalize:
        mean = train.mean(axis=0, keepdims=True)
        std  = train.std( axis=0, keepdims=True) + 1e-8
        train = (train - mean) / std
        test  = (test  - mean) / std

    print(f"[Data] {name} | 通道数={train.shape[1]} "
          f"| 训练={train.shape[0]:,} | 测试={test.shape[0]:,} "
          f"| 异常率={labels.mean():.2%}")

    return train, test, labels


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset（滑动窗口）
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    滑动窗口数据集。

    样本：
      x: (window_size, N)  — 历史窗口
      y: (N,)              — 下一时间步（预测目标）
    """

    def __init__(self, data: np.ndarray, window_size: int, step: int = 1):
        super().__init__()
        self.data        = torch.from_numpy(data)
        self.window_size = window_size
        self.indices     = list(range(0, len(data) - window_size, step))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.indices[idx]
        x = self.data[s : s + self.window_size]   # (T, N)
        y = self.data[s + self.window_size]        # (N,)
        return x, y

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# 构建 DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(
    data_dir: str,
    dataset: str = "smap",
    window_size: int = 100,
    train_step: int = 5,
    test_step: int = 1,
    batch_size: int = 64,
    normalize_data: bool = True,
    num_workers: int = 0,
    # 兼容旧接口（忽略）
    label_file: str = None,
) -> Tuple[DataLoader, DataLoader, np.ndarray, int]:
    """
    Returns:
        train_loader, test_loader, test_labels (T,), n_channels
    """
    train, test, labels = load_data(data_dir, dataset, normalize_data)

    train_ds = TimeSeriesDataset(train, window_size, step=train_step)
    test_ds  = TimeSeriesDataset(test,  window_size, step=test_step)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0),
    )

    return train_loader, test_loader, labels, train.shape[1]
