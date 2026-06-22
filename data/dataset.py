"""
数据加载模块 — 兼容 Telemanom 的 SMAP / MSL 格式

Telemanom 数据目录结构：
  data/
    train/  {chan_id}.npy   shape: (timesteps, n_features)
    test/   {chan_id}.npy   shape: (timesteps, n_features)
  labeled_anomalies.csv   列: chan_id, spacecraft, anomaly_sequences, ...

下载方式：
  git clone https://github.com/khundman/telemanom
  数据文件在 telemanom/data/ 下，将整个 data/ 文件夹复制到本项目 datasets/ 目录
"""

from __future__ import annotations

import ast
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# 通道列表（SMAP / MSL）
# ─────────────────────────────────────────────────────────────────────────────

SMAP_CHANNELS = [
    "P-1","S-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7","E-8","E-9",
    "E-10","E-11","E-12","E-13","A-1","D-1","P-2","P-3","D-2","D-3","D-4",
    "A-2","A-3","A-4","G-1","G-2","D-5","D-6","D-7","F-1","P-4","G-3",
    "T-1","T-2","D-8","D-9","F-2","G-4","T-3","D-11","D-12","B-1","G-6",
    "G-7","F-3","D-13","P-7","R-1","A-5","A-6","A-7","D-14","D-15","D-16",
]

MSL_CHANNELS = [
    "M-6","M-1","M-2","S-2","P-10","T-4","T-5","F-7","M-3","M-4","M-5",
    "P-15","C-1","C-2","T-12","T-13","F-4","F-5","D-16","M-7","F-6","T-9",
    "P-11","D-9","T-8","D-5","F-1",
]

CHANNEL_MAP = {"smap": SMAP_CHANNELS, "msl": MSL_CHANNELS}


# ─────────────────────────────────────────────────────────────────────────────
# 数据预处理工具
# ─────────────────────────────────────────────────────────────────────────────

def _load_channel(data_dir: str, chan_id: str, split: str) -> np.ndarray:
    """
    加载单个通道的 .npy 文件。
    返回第一维特征（主要遥测值），shape: (timesteps,)
    """
    path = os.path.join(data_dir, split, f"{chan_id}.npy")
    arr = np.load(path)  # (T, n_features)
    return arr[:, 0].astype(np.float32)  # 取第一个特征作为主遥测值


def load_multivariate(data_dir: str, dataset: str, split: str) -> np.ndarray:
    """
    将所有通道堆叠为多变量时间序列。
    返回 shape: (timesteps, N_channels)
    """
    channels = CHANNEL_MAP[dataset]
    series = []
    for ch in channels:
        try:
            s = _load_channel(data_dir, ch, split)
            series.append(s)
        except FileNotFoundError:
            print(f"[WARN] 通道 {ch} 文件不存在，跳过")
    # 对齐最短长度（不同通道时间步可能略有差异）
    min_len = min(s.shape[0] for s in series)
    stacked = np.stack([s[:min_len] for s in series], axis=1)  # (T, N)
    return stacked


def load_labels(label_file: str, dataset: str) -> dict:
    """
    从 labeled_anomalies.csv 解析异常区间。
    返回 {chan_id: [(start, end), ...]} 字典（测试集相对索引）
    """
    df = pd.read_csv(label_file)
    df = df[df["spacecraft"].str.lower() == dataset.lower()]
    labels = {}
    for _, row in df.iterrows():
        chan = row["chan_id"]
        seqs = ast.literal_eval(row["anomaly_sequences"])  # 列表 of [start, end]
        labels[chan] = [(int(s[0]), int(s[1])) for s in seqs]
    return labels


def build_test_labels(channels: list, labels: dict, test_len: int) -> np.ndarray:
    """
    构建与测试集等长的逐点标签向量 (test_len, N_channels)。
    1 = 异常，0 = 正常
    """
    N = len(channels)
    label_arr = np.zeros((test_len, N), dtype=np.float32)
    for i, chan in enumerate(channels):
        if chan in labels:
            for start, end in labels[chan]:
                s = min(start, test_len - 1)
                e = min(end, test_len)
                label_arr[s:e, i] = 1.0
    return label_arr


def normalize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    用训练集均值/标准差对训练集和测试集做 z-score 归一化。
    """
    mean = train.mean(axis=0, keepdims=True)
    std  = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mean) / std, (test - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SMAPDataset(Dataset):
    """
    滑动窗口数据集。

    每个样本：
      x:  (window_size, N_channels)  — 历史窗口
      y:  (N_channels,)              — 下一时间步（预测目标）
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 100,
        step: int = 1,
    ):
        super().__init__()
        self.data = torch.from_numpy(data)      # (T, N)
        self.window_size = window_size
        self.step = step
        self.indices = list(range(0, len(data) - window_size, step))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        x = self.data[start : start + self.window_size]        # (T, N)
        y = self.data[start + self.window_size]                # (N,)
        return x, y

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# 便捷函数：一次性构建训练 / 测试 DataLoader
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
    num_workers: int = 0,       # Windows 下建议设 0（避免多进程问题）
) -> Tuple[DataLoader, DataLoader, np.ndarray, list]:
    """
    返回 (train_loader, test_loader, test_labels, channels)

    test_labels: (T_test, N_channels)  — 逐点异常标签 (0/1)
    channels:    通道名称列表（和数据列顺序一致）
    """
    channels = CHANNEL_MAP[dataset]

    # ── 加载原始数据 ──
    train_data = load_multivariate(data_dir, dataset, "train")
    test_data  = load_multivariate(data_dir, dataset, "test")

    # 通道对齐：只保留两者都有的通道
    available = []
    train_list, test_list = [], []
    for i, ch in enumerate(channels):
        if i < train_data.shape[1] and i < test_data.shape[1]:
            available.append(ch)
            train_list.append(train_data[:, i])
            test_list.append(test_data[:, i])
    channels = available
    train_data = np.stack(train_list, axis=1)
    test_data  = np.stack(test_list,  axis=1)

    # ── 归一化 ──
    if normalize_data:
        train_data, test_data = normalize(train_data, test_data)

    # ── 标签 ──
    labels_dict = {}
    if label_file and os.path.exists(label_file):
        labels_dict = load_labels(label_file, dataset)
    test_labels = build_test_labels(channels, labels_dict, len(test_data))

    # ── 构建 Dataset / DataLoader ──
    train_ds = SMAPDataset(train_data, window_size, step=train_step)
    test_ds  = SMAPDataset(test_data,  window_size, step=test_step)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    print(f"[Data] {dataset.upper()} | 通道数={len(channels)} "
          f"| 训练={len(train_ds)} 样本 | 测试={len(test_ds)} 样本")

    return train_loader, test_loader, test_labels, channels
