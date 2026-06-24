"""
MambGAT-AD — 完整模型

整体架构：

  输入 X: (B, T, N)         — 多变量遥测序列
       │
  [输入嵌入]               — Linear: N → N×D
       │
  [ST-Mamba-GAT Encoder]   — n_blocks 个时空耦合块
       │
  [预测头]                 — 预测下一时间步 X̂: (B, N)
       │
  [残差计算]               — E = |X_last - X̂|: (B, N)
       │
  [异常分数]               — 返回 (B, N) 残差向量，供阈值模块使用

训练目标：最小化预测误差（MSE loss）
推理时：用残差大小作为异常分数，通过动态阈值判断异常点
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn

from .st_block import STMambaGATEncoder


class MambGATAD(nn.Module):
    """
    MambGAT-AD: Spatiotemporal Mamba with Graph Attention
    for Spacecraft Telemetry Anomaly Detection

    Args:
        n_channels:  传感器/通道数（= 图节点数）
        window_size: 输入时间窗口长度 T
        d_model:     特征嵌入维度
        n_blocks:    ST-Mamba-GAT 块数量
        n_heads:     GAT 注意力头数
        d_state:     Mamba SSM 状态维度
        d_conv:      Mamba 局部卷积核大小
        expand:      Mamba 内部维度倍数
        pred_len:    预测步数（默认 1 = 预测下一步）
        dropout:     dropout 概率
        top_k:       GAT 每节点保留邻居数（None = 全图）
    """

    def __init__(
        self,
        n_channels: int,
        window_size: int,
        d_model: int = 64,
        n_blocks: int = 2,
        n_heads: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        pred_len: int = 1,
        dropout: float = 0.1,
        top_k: int = None,
    ):
        super().__init__()
        self.n_channels  = n_channels
        self.window_size = window_size
        self.d_model     = d_model
        self.pred_len    = pred_len

        # ── 1. 输入嵌入（每个通道的原始值 → D 维特征）──────────────
        # 每个节点共享同一嵌入层（参数效率高）
        self.input_proj = nn.Linear(1, d_model)

        # ── 2. 位置编码（可选，帮助 Mamba 感知时序位置）────────────
        self.pos_emb = nn.Parameter(
            torch.randn(1, window_size, 1, d_model) * 0.02
        )

        # ── 3. 时空编码器（核心模块）────────────────────────────────
        self.encoder = STMambaGATEncoder(
            d_model=d_model,
            n_nodes=n_channels,
            n_blocks=n_blocks,
            n_heads=n_heads,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            top_k=top_k,
        )

        # ── 4. 预测头（编码器最后一步 → 预测未来 pred_len 步）──────
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, N)  — 滑动窗口输入，N 个通道

        Returns:
            pred:  (B, N, pred_len)  — 预测值
            score: (B, N)            — 异常分数（最后一步的预测残差绝对值）
        """
        B, T, N = x.shape

        # 1. 输入嵌入：(B, T, N) → (B, T, N, D)
        h = x.unsqueeze(-1)                    # (B, T, N, 1)
        h = self.input_proj(h)                 # (B, T, N, D)
        h = h + self.pos_emb[:, :T]            # 加位置编码

        # 2. 时空编码：(B, T, N, D) → (B, T, N, D)
        h = self.encoder(h)

        # 3. 预测：取最后一个时间步做预测
        last = h[:, -1, :, :]                  # (B, N, D)
        pred = self.pred_head(last)             # (B, N, pred_len)

        # 4. 异常分数：预测值 vs 实际值（最后一步）的 MAE
        #    （训练时不用 score，推理时使用）
        actual_last = x[:, -1, :]              # (B, N)
        score = (pred.squeeze(-1) - actual_last).abs()   # (B, N)

        return pred, score

    # ─────────────────────────────────────────────────────────────────
    def get_graph(self, head_idx: int = 0) -> torch.Tensor:
        """
        获取学习到的传感器耦合图（用于论文可视化）。
        返回 (N, N) 邻接矩阵
        """
        return self.encoder.blocks[0].spatial.get_adjacency(head_idx=head_idx)

    # ─────────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 损失函数
# ─────────────────────────────────────────────────────────────────────────────

class PredictionLoss(nn.Module):
    """预测损失：MSE + 可选的 L1 正则（稀疏化图结构）"""

    def __init__(self, alpha: float = 0.5):
        """alpha: L1 和 MSE 的混合比例"""
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(
        self,
        pred: torch.Tensor,   # (B, N, pred_len)
        target: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        # target 扩展为 (B, N, pred_len)
        target_exp = target.unsqueeze(-1).expand_as(pred)
        loss_mse = self.mse(pred, target_exp)
        loss_mae = self.mae(pred, target_exp)
        return (1 - self.alpha) * loss_mse + self.alpha * loss_mae
