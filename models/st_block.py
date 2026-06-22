"""
ST-Mamba-GAT Block — 时空耦合核心模块（论文最核心的创新点）

设计思路：
  时间维度：Mamba SSM 沿 T 轴扫描，捕捉长程时序依赖（线性复杂度）
  空间维度：GAT 沿 N 轴聚合，捕捉传感器间物理耦合关系
  耦合方式：空间上下文作为额外输入增强 Mamba 的时序建模
            （而非简单串联，空间信息被"注入"到时序状态更新中）

完整数据流（单个 ST-Mamba-GAT Block）：

  输入 X: (B, T, N, D)
       │
       ├─ 空间路径：对每个 t，GAT(X[:,t,:,:]) → Z_spatial (B, T, N, D)
       │
       ├─ 空间增强：X_enhanced = X + α * Z_spatial
       │
       └─ 时序路径：对每个 n，Mamba(X_enhanced[:,:,n,:]) → Y (B, T, N, D)
       │
       └─ 输出 = X + Y  （残差）

多块堆叠时，空间与时序特征交替强化，逐渐发现复杂时空耦合模式。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ssm_layer import MambaBlock
from .gat_layer import GraphModule


class STMambaGATBlock(nn.Module):
    """
    单个时空 Mamba-GAT 块。

    Args:
        d_model:    特征维度
        n_nodes:    节点数（传感器/通道数）
        n_heads:    GAT 注意力头数
        d_state:    Mamba SSM 状态维度
        d_conv:     Mamba 局部卷积核大小
        expand:     Mamba 内部维度倍数
        dropout:    dropout 概率
        top_k:      GAT 稀疏化参数（None = 全连接图）
    """

    def __init__(
        self,
        d_model: int,
        n_nodes: int,
        n_heads: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        top_k: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_nodes = n_nodes

        # ── 空间路径：GAT（每个时间步做一次图聚合）─────────────────
        self.spatial = GraphModule(
            d_model=d_model, n_nodes=n_nodes,
            n_heads=n_heads, n_layers=1,
            dropout=dropout, top_k=top_k,
        )

        # ── 空间注入权重（可学习标量，控制空间上下文的贡献）────────
        self.spatial_gate = nn.Parameter(torch.zeros(1))

        # ── 时序路径：Mamba（每个节点做一次时序扫描）───────────────
        self.temporal = MambaBlock(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand, dropout=dropout,
        )

        # ── 融合后的前馈网络（FFN）─────────────────────────────────
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(d_model)

    # ─────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, D)
        Returns:
            out: (B, T, N, D)
        """
        B, T, N, D = x.shape

        # ── 1. 空间路径（对每个时间步做 GAT）────────────────────────
        # 将 T 合并进 batch 维，批量处理所有时间步
        x_flat = x.reshape(B * T, N, D)          # (B*T, N, D)
        z_spatial = self.spatial(x_flat)          # (B*T, N, D)
        z_spatial = z_spatial.reshape(B, T, N, D) # (B, T, N, D)

        # 空间信息以可学习权重注入（避免过强的空间信号淹没时序信号）
        x_enhanced = x + torch.sigmoid(self.spatial_gate) * z_spatial

        # ── 2. 时序路径（对每个节点做 Mamba 扫描）───────────────────
        # 将 N 合并进 batch 维，批量处理所有节点
        x_t = x_enhanced.permute(0, 2, 1, 3)     # (B, N, T, D)
        x_t = x_t.reshape(B * N, T, D)            # (B*N, T, D)
        y_t = self.temporal(x_t)                  # (B*N, T, D)
        y_t = y_t.reshape(B, N, T, D)             # (B, N, T, D)
        y = y_t.permute(0, 2, 1, 3)               # (B, T, N, D)

        # ── 3. 残差融合 + FFN ─────────────────────────────────────
        out = x + y
        out = out + self.ffn(out)
        return self.norm_out(out)


# ─────────────────────────────────────────────────────────────────────────────
# 多块堆叠
# ─────────────────────────────────────────────────────────────────────────────

class STMambaGATEncoder(nn.Module):
    """
    堆叠 n_blocks 个 STMambaGATBlock，构成完整的时空编码器。

    可以把这个模块理解为：
      "专为航天器遥测数据设计的时空特征提取器"
    """

    def __init__(
        self,
        d_model: int,
        n_nodes: int,
        n_blocks: int = 2,
        n_heads: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        top_k: int = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            STMambaGATBlock(
                d_model=d_model, n_nodes=n_nodes, n_heads=n_heads,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, top_k=top_k,
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N, D) → (B, T, N, D)"""
        for block in self.blocks:
            x = block(x)
        return x
