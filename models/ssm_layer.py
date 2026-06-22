"""
SelectiveSSM — 纯 PyTorch 实现的 Mamba 状态空间模型

设计目标：
  • Windows + Anaconda 直接可用（无需编译 CUDA 算子）
  • 与 mamba_ssm.Mamba 接口兼容，Linux 上可直接替换以获得 ~10x 加速
  • 复现 Mamba 论文的核心机制：选择性状态空间、SiLU 门控

参考文献：
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces", ICLR 2024.  arXiv:2312.00752
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    选择性状态空间模型（Mamba-like），纯 PyTorch 实现。

    核心公式（离散化后）：
        h_t = A_t ⊙ h_{t-1}  +  B_t ⊙ x_t
        y_t = C_t ⊙ h_t  +  D ⊙ x_t

    其中 A_t, B_t, C_t 均为**输入相关**（selective），
    这是 Mamba 相对 S4 的核心创新。

    Args:
        d_model:  输入 / 输出维度
        d_state:  SSM 状态维度（越大捕捉历史越长，默认 16）
        d_conv:   局部 depthwise 卷积核大小（default 4）
        expand:   内部维度倍数（d_inner = expand * d_model）
        dt_rank:  时间步长投影秩，'auto' 时取 ceil(d_model/16)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[str, int] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        # ── 输入投影（x 分支 + z 门控分支）─────────────────────────────
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # ── 局部 depthwise 卷积（捕捉短程依赖）───────────────────────
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # ── SSM 参数：将 x 映射到 Δ(dt), B, C ──────────────────────
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        # Δ 投影（时间步长，控制"选择性"）
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # ── 初始化 dt_proj.bias（对数均匀分布）───────────────────────
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus 反函数
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # ── A 矩阵（HiPPO 初始化，log 参数化保证负值）──────────────
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)          # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # ── D（跳跃连接权重）────────────────────────────────────────
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # ── 输出投影 ─────────────────────────────────────────────────
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            y: (B, T, d_model)
        """
        B, T, _ = x.shape

        # 1. 输入投影：产生 x 分支和 z 门控
        xz = self.in_proj(x)                          # (B, T, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                 # 各 (B, T, d_inner)

        # 2. 局部卷积（因果，裁掉多余 padding）
        x_in = x_in.transpose(1, 2)                   # (B, d_inner, T)
        x_in = self.conv1d(x_in)[..., :T]             # 因果截断
        x_in = x_in.transpose(1, 2)                   # (B, T, d_inner)
        x_in = F.silu(x_in)

        # 3. 计算输入相关参数 Δ, B_ssm, C_ssm
        ssm_params = self.x_proj(x_in)                # (B, T, dt_rank + 2*d_state)
        dt_raw, B_ssm, C_ssm = torch.split(
            ssm_params,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        # Δ (离散时间步长，控制信息保留量)
        dt = F.softplus(self.dt_proj(dt_raw))          # (B, T, d_inner)
        # A（连续，负值）
        A = -torch.exp(self.A_log.float())             # (d_inner, d_state)

        # 4. 离散化（零阶保持 ZOH）
        #    A_disc[t] = exp(Δ[t] * A)
        #    B_disc[t] = Δ[t] * B_ssm[t]
        dA = torch.exp(
            dt.unsqueeze(-1) * A                       # (B,T,d_inner,d_state)
        )
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(-2)   # (B,T,d_inner,d_state)

        # 5. 顺序扫描（recurrent scan）
        #    h_t = A_disc_t ⊙ h_{t-1} + B_disc_t ⊙ x_t
        #    y_t = sum(C_t ⊙ h_t, dim=-1)
        h = x_in.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * x_in[:, t].unsqueeze(-1)
            y_t = (h * C_ssm[:, t].unsqueeze(-2)).sum(-1)   # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                     # (B, T, d_inner)

        # 6. D 项（跳跃连接）+ SiLU 门控
        y = y + self.D * x_in
        y = y * F.silu(z)

        # 7. 输出投影
        return self.out_proj(y)                        # (B, T, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# MambaBlock：完整的 Mamba 块（含 LayerNorm + 残差）
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    单个 Mamba 块：LayerNorm → SelectiveSSM → 残差

    与 Transformer Block 地位等价，可堆叠。
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm  = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return x + self.drop(self.ssm(self.norm(x)))
