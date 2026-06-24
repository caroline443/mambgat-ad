"""
数据加载模块 — 支持 Telemanom / Kaggle 格式

数据目录结构（Kaggle: patrickfleith/nasa-anomaly-detection-dataset-smap-msl）：
  datasets/archive/
    data/
      train/   A-1.npy  P-1.npy  ...（SMAP + MSL 混在一起）
      test/    A-1.npy  P-1.npy  ...
    labeled_anomalies.csv
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
# 通道发现
# ─────────────────────────────────────────────────────────────────────────────

def get_channels(label_file: str, dataset: str) -> List[str]:
    df = pd.read_csv(label_file, encoding='utf-8')
    mask = df["spacecraft"].str.upper() == dataset.upper()
    return df.loc[mask, "chan_id"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 单通道加载
# ─────────────────────────────────────────────────────────────────────────────

def _load_channel(data_dir: str, split: str, chan_id: str) -> np.ndarray:
    path = os.path.join(data_dir, split, f"{chan_id}.npy")
    arr = np.load(path)
    return arr[:, 0].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 多通道加载（过滤过短通道）
# ─────────────────────────────────────────────────────────────────────────────

def load_multivariate(
    data_dir: str,
    split: str,
    channels: List[str],
    min_len: int = 1000,
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
        s = np.where(np.isfinite(s), s, 0.0)
        series.append(s)
        valid.append(ch)

    if skipped:
        print(f"  [跳过] {len(skipped)} 个过短通道: "
              f"{[f'{c}({l})' for c, l in skipped[:5]]}"
              f"{'...' if len(skipped) > 5 else ''}")

    align_len = min(s.shape[0] for s in series)
    data = np.stack([s[:align_len] for s in series], axis=1)
    return data, valid


# ─────────────────────────────────────────────────────────────────────────────
# 归一化
# ─────────────────────────────────────────────────────────────────────────────

def normalize(train: np.ndarray, test: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std  = train.std(axis=0,  keepdims=True)
    std  = np.where(std < 1e-4, 1.0, std)
    train_n = np.clip((train - mean) / std, -10.0, 10.0)
    test_n  = np.clip((test  - mean) / std, -10.0, 10.0)
    return train_n.astype(np.float32), test_n.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 逐通道标签（返回 (T, N)，每通道独立评估用）
# ─────────────────────────────────────────────────────────────────────────────

def build_labels_per_channel(
    label_file: str,
    channels: List[str],
    test_len: int,
) -> np.ndarray:
    """
    返回 (test_len, N_channels) 逐通道二值标签。
    每个通道有自己的异常窗口，不做 OR 合并。
    这是标准的 SMAP 评估协议（类 Telemanom）。
    """
    df = pd.read_csv(label_file, encoding='utf-8')
    labels = np.zeros((test_len, len(channels)), dtype=np.int32)
    for i, ch in enumerate(channels):
        row = df[df["chan_id"] == ch]
        if row.empty:
            continue
        seqs = ast.literal_eval(row.iloc[0]["anomaly_sequences"])
        for start, end in seqs:
            s = min(int(start), test_len - 1)
            e = min(int(end), test_len)
            labels[s:e, i] = 1
    return labels


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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.indices[idx]
        return self.data[s : s + self.ws], self.data[s + self.ws]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# 构建 DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(
    data_dir: str,
    label_file: str,
    dataset: str = "smap",
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
        test_labels: (T_test, N_channels) 逐通道标签,
        n_channels: int
    """
    channels = get_channels(label_file, dataset)
    print(f"[Data] {dataset.upper()} | CSV 中共 {len(channels)} 个通道")

    train_raw, channels = load_multivariate(data_dir, "train", channels)
    test_raw,  _        = load_multivariate(data_dir, "test",  channels)
    n_channels = len(channels)

    if normalize_data:
        train_raw, test_raw = normalize(train_raw, test_raw)

    # 逐通道标签（不 OR 合并）
    test_labels = build_labels_per_channel(label_file, channels, len(test_raw))
    global_anomaly_rate = test_labels.any(axis=1).mean()

    print(f"[Data] 通道数={n_channels} | "
          f"训练={len(train_raw) - window_size:,} | "
          f"测试={len(test_raw) - window_size:,} | "
          f"全局异常率={global_anomaly_rate:.2%} "
          f"(OR逻辑，per-channel 平均={test_labels.mean():.2%})")

    train_ds = TimeSeriesDataset(train_raw, window_size, step=train_step)
    test_ds  = TimeSeriesDataset(test_raw,  window_size, step=test_step)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(num_workers > 0),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0),
    )

    return train_loader, test_loader, test_labels, n_channels
