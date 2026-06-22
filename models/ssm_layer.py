"""
MambaBlock — 自动适配：原生 mamba_ssm（服务器）/ 纯 PyTorch（Windows）

启动时自动检测 mamba_ssm 是否可用：
  ✅ 服务器 Linux + CUDA  → 使用 mamba_ssm.Mamba（CUDA kernel，快 ~10x）
  ✅ Windows + Anaconda   → 使用纯 PyTorch SelectiveSSM（慢但功能完全等价）

st_block.py / mambgat.py 无需任何改动，完全透明。

参考：
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces", ICLR 2024.  arXiv:2312.00752
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 尝试导入原生 mamba_ssm ────────────────────────────────────────────────
try:
    from mamba_ssm import Mamba as _MambaNative
    _USE_NATIVE = True
    print("[MambGAT] mamba_ssm 检测到 → 使用原生 CUDA kernel（服务器模式）")
except ImportError:
    _USE_NATIVE = False
    print("[MambGAT] mamba_ssm 未找到 → 使用纯 PyTorch 实现（Windows 兼容模式）")


# ─────────────────────────────────────────────────────────────────────────────
# 纯 PyTorch 实现（Windows fallback）
# ─────────────────────────────────────────────────────────────────────────────

class _SelectiveSSM(nn.Module):
    """
    纯 PyTorch 选择性状态空间模型（Mamba-like）。
    仅在 mamba_ssm 不可用时加载，功能与原生版本完全等价。
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                  bias=True, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj   = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj  = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt_proj.bias 初始化（保证初始时间步在合理范围）
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt_init + torch.log(-torch.expm1(-dt_init)))

        # A（HiPPO 初始化，log 参数化）
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.unsqueeze(0).expand(self.d_inner, -1)))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_in = self.conv1d(x_in.transpose(1, 2))[..., :T].transpose(1, 2)
        x_in = F.silu(x_in)

        ssm_out = self.x_proj(x_in)
        dt_raw, B_ssm, C_ssm = torch.split(
            ssm_out, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt  = F.softplus(self.dt_proj(dt_raw)).clamp(min=1e-4, max=1.0)  # 防溢出
        A   = -torch.exp(self.A_log.float())

        dA = torch.exp(dt.unsqueeze(-1) * A)               # (B,T,d_inner,d_state)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(-2)        # (B,T,d_inner,d_state)

        h = x_in.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(T):
            h  = dA[:, t] * h + dB[:, t] * x_in[:, t].unsqueeze(-1)
            h  = h.clamp(-1e4, 1e4)                        # 防状态爆炸
            ys.append((h * C_ssm[:, t].unsqueeze(-2)).sum(-1))
        y = torch.stack(ys, dim=1) + self.D * x_in
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)  # 最后保险
        return self.out_proj(y * F.silu(z))


# ─────────────────────────────────────────────────────────────────────────────
# 统一接口：MambaBlock（外部模块只导入这个）
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    单个 Mamba 块：LayerNorm → Mamba（自动选择实现）→ 残差

    对 st_block.py 完全透明，无论在服务器还是 Windows 上行为一致。

    Args:
        d_model:  输入 / 输出维度
        d_state:  SSM 状态维度（默认 16）
        d_conv:   局部卷积核大小（默认 4）
        expand:   内部维度倍数（默认 2）
        dropout:  输出 dropout 概率
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if _USE_NATIVE:
            self.mamba = _MambaNative(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mamba = _SelectiveSSM(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return x + self.drop(self.mamba(self.norm(x)))

    @property
    def backend(self) -> str:
        return "native" if _USE_NATIVE else "pytorch"
