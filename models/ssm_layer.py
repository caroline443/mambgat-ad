"""
MambaBlock — 直接调用原生 mamba_ssm（服务器 Linux + CUDA 环境）

依赖：
  pip install mamba-ssm causal-conv1d --no-build-isolation
  要求：Linux, NVIDIA GPU, CUDA >= 11.6, PyTorch >= 2.0

原生 mamba_ssm 使用 CUDA selective scan 算子，比纯 PyTorch 实现快 ~10x，
且显存占用更低（fused kernel 减少中间激活值存储）。

参考：
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces", ICLR 2024.  arXiv:2312.00752
  https://github.com/state-spaces/mamba
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """
    单个 Mamba 块：LayerNorm → Mamba → 残差

    直接包装 mamba_ssm.Mamba，接口与之前的纯 PyTorch 版本完全一致，
    st_block.py 无需任何改动。

    Args:
        d_model:  输入 / 输出维度
        d_state:  SSM 状态维度（越大历史记忆越长，默认 16）
        d_conv:   局部 depthwise 卷积核大小（默认 4）
        expand:   内部维度倍数（d_inner = expand * d_model，默认 2）
        dropout:  输出 dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            y: (B, T, d_model)
        """
        return x + self.drop(self.mamba(self.norm(x)))
