"""
数据加载模块 — 支持两种 SMAP 格式

格式A（Anomaly Transformer，主流论文标准）：
  datasets/AT/
    SMAP_train.npy        (T_train, N)
    SMAP_test.npy         (T_test,  N)
    SMAP_test_label.npy   (T_test,) 全局标签，异常率~13%
  下载：https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm

格式B（Telemanom/Kaggle，逐通道评估）：
  datasets/archive/
    data/train/*.npy  data/test/*.npy
    labeled_anomalies.csv

配置 data.format: "AT" 或 "telemanom"
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
# 格式A：Anomaly Transformer 单文件格式（主流论文标准）
# ─────────────────────────────────────────────────────────────────────────────

def load_at_format(
    data_dir: str,
    dataset: str,       # "smap" | "msl"
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 Anomaly Transformer 格式数据。
    返回 (train, test, global_labels)，标签为 (T_test,) 全局标签。
    """
    name = dataset.upper()
    train = np.load(os.path.join(data_dir, f"{name}_train.npy")).astype(np.float32)
    test  = np.load(os.path.join(data_dir, f"{name}_test.npy" )).astype(np.float32)
    label = np.load(os.path.join(data_dir, f"{name}_test_label.npy")).astype(np.int32)

    if normalize:
        mean = train.mean(0, keepdims=True)
        std  = np.where(train.std(0, keepdims=True) < 1e-4, 1.0,
                        train.std(0, keepdims=True))
        train = np.clip((train - mean) / std, -10, 10).astype(np.float32)
        test  = np.clip((test  - mean) / std, -10, 10).astype(np.float32)

    print(f"[Data] {name}(AT格式) | 通道数={train.shape[1]} "
          f"| 训练={len(train):,} | 测试={len(test):,} "
          f"| 异常率={label.mean():.2%}")
    return train, test, label


# ─────────────────────────────────────────────────────────────────────────────
# 格式B：Telemanom 逐通道格式（Kaggle 数据集）
# ─────────────────────────────────────────────────────────────────────────────

def get_channels(label_file: str, dataset: str) -> List[str]:
    df = pd.read_csv(label_file, encoding='utf-8')
    return df.loc[df["spacecraft"].str.upper() == dataset.upper(),
                  "chan_id"].tolist()


def _load_channel(data_dir: str, split: str, chan_id: str) -> np.ndarray:
    arr = np.load(os.path.join(data_dir, split, f"{chan_id}.npy"))
    return arr[:, 0].astype(np.float32)


def load_multivariate(
    data_dir: str, split: str, channels: List[str], min_len: int = 1000,
) -> Tuple[np.ndarray, List[str]]:
    series, valid, skipped = [], [], []
    for ch in channels:
        path = os.path.join(data_dir, split, f"{ch}.npy")
        if not os.path.exists(path):
            continue
        s = _load_channel(data_dir, split, ch)
        if s.shape[0] < min_len:
            skipped.append((ch, s.shape[0]))
            continue
        series.append(np.where(np.isfinite(s), s, 0.0))
        valid.append(ch)
    if skipped:
        print(f"  [跳过] {len(skipped)} 个过短通道: "
              f"{[f'{c}({l})' for c, l in skipped[:5]]}"
              f"{'...' if len(skipped)>5 else ''}")
    align = min(s.shape[0] for s in series)
    return np.stack([s[:align] for s in series], axis=1), valid


def build_labels_per_channel(
    label_file: str, channels: List[str], test_len: int,
) -> np.ndarray:
    """返回 (T, N) 逐通道标签"""
    df = pd.read_csv(label_file, encoding='utf-8')
    labels = np.zeros((test_len, len(channels)), dtype=np.int32)
    for i, ch in enumerate(channels):
        row = df[df["chan_id"] == ch]
        if row.empty:
            continue
        for s, e in ast.literal_eval(row.iloc[0]["anomaly_sequences"]):
            labels[min(int(s), test_len-1):min(int(e), test_len), i] = 1
    return labels


def load_telemanom_format(
    data_dir: str, label_file: str, dataset: str, normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    加载 Telemanom 格式，返回 (train, test, per_channel_labels, channels)
    per_channel_labels: (T_test, N)
    """
    channels = get_channels(label_file, dataset)
    print(f"[Data] {dataset.upper()}(Telemanom) | CSV中 {len(channels)} 通道")
    train, channels = load_multivariate(data_dir, "train", channels)
    test,  _        = load_multivariate(data_dir, "test",  channels)
    if normalize:
        mean = train.mean(0, keepdims=True)
        std  = np.where(train.std(0, keepdims=True) < 1e-4, 1.0,
                        train.std(0, keepdims=True))
        train = np.clip((train - mean) / std, -10, 10).astype(np.float32)
        test  = np.clip((test  - mean) / std, -10, 10).astype(np.float32)
    labels = build_labels_per_channel(label_file, channels, len(test))
    print(f"[Data] 通道数={len(channels)} | 训练={len(train):,} | "
          f"测试={len(test):,} | 全局异常率(OR)={labels.any(1).mean():.2%} "
          f"| 逐通道平均={labels.mean():.2%}")
    return train, test, labels, channels


# ─────────────────────────────────────────────────────────────────────────────
# 通用归一化
# ─────────────────────────────────────────────────────────────────────────────

def normalize(train: np.ndarray, test: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(0, keepdims=True)
    std  = np.where(train.std(0, keepdims=True) < 1e-4, 1.0,
                    train.std(0, keepdims=True))
    return (np.clip((train - mean) / std, -10, 10).astype(np.float32),
            np.clip((test  - mean) / std, -10, 10).astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, window_size: int, step: int = 1):
        super().__init__()
        self.data    = torch.from_numpy(data)
        self.ws      = window_size
        self.indices = list(range(0, len(data) - window_size, step))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        return self.data[s:s+self.ws], self.data[s+self.ws]

    @property
    def n_channels(self):
        return self.data.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(
    data_dir: str,
    dataset: str = "smap",
    fmt: str = "AT",           # "AT" | "telemanom"
    label_file: str = None,    # telemanom 格式需要
    window_size: int = 100,
    train_step: int = 1,
    test_step: int = 1,
    batch_size: int = 64,
    normalize_data: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, np.ndarray, int]:
    """
    Returns:
        train_loader, test_loader,
        test_labels: AT格式=(T,) 全局标签; telemanom格式=(T,N) 逐通道标签
        n_channels: int
    """
    if fmt.upper() == "AT":
        train, test, labels = load_at_format(data_dir, dataset, normalize_data)
        n_ch = train.shape[1]
    else:
        assert label_file, "telemanom 格式需要 label_file"
        train, test, labels, _ = load_telemanom_format(
            data_dir, label_file, dataset, normalize_data)
        n_ch = train.shape[1]

    train_ds = TimeSeriesDataset(train, window_size, train_step)
    test_ds  = TimeSeriesDataset(test,  window_size, test_step)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False,
                              pin_memory=(num_workers > 0))
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers,
                              pin_memory=(num_workers > 0))
    return train_loader, test_loader, labels, n_ch
