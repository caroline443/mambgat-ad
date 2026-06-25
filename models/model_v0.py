"""
MambGAT-AD v0 — 基于官方 mamba-ssm 的最小基线

架构（故意保持极简，每个组件都清晰可追溯）：

  输入 X: (B, T, N)
       │
  [Linear Embed]      每个通道独立投影到 D 维  →  (B*N, T, D)
       │
  [Mamba Encoder]     官方 mamba-ssm，沿 T 轴扫描（通道间独立）
  (B*N, T, D)
       │
  [Pred Head]         取最后时间步 → 预测下一步  →  (B, N, 1)
       │
  [Recon Head]        所有时间步 → 重建输入      →  (B, T, N)
       │
  [Anomaly Score]     pred_err + recon_err       →  (B, N)

损失：
  L = MSE(pred, y) + beta * MSE(recon, x)

这个版本故意不加 GAT、不加多尺度、不加对比损失，
目的是建立干净的基线，确认数据流和评估流程正确。
后续版本在此基础上逐步叠加组件。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from mamba_ssm import Mamba


# ─────────────────────────────────────────────────────────────────────────────
# Mamba 编码器（堆叠 n_blocks 个官方 Mamba Block）
# ─────────────────────────────────────────────────────────────────────────────

class MambaEncoder(nn.Module):
    """
    堆叠 n_blocks 个 Mamba Block，每块后接 LayerNorm + 残差。

    输入/输出：(B, T, D)
    """

    def __init__(
        self,
        d_model: int,
        n_blocks: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                ),
                nn.Dropout(dropout),
            )
            for _ in range(n_blocks)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        for block, norm in zip(self.blocks, self.norms):
            x = norm(x + block(x))   # Pre-norm 残差
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 主模型 v0
# ─────────────────────────────────────────────────────────────────────────────

class MambGATAD(nn.Module):
    """
    MambGAT-AD v0：最小基线（Linear Embed + Mamba + Pred/Recon Head）

    Args:
        n_channels:  传感器数量 N
        window_size: 输入窗口长度 T
        d_model:     嵌入维度 D
        n_blocks:    Mamba 块数量
        d_state:     SSM 状态维度
        d_conv:      局部卷积核大小
        expand:      内部扩展倍数
        pred_len:    预测步数（默认 1）
        dropout:     dropout 概率
        # 以下参数 v0 不使用，保留接口兼容性供后续版本
        n_heads, top_k, patch_sizes, n_snapshots
    """

    def __init__(
        self,
        n_channels: int,
        window_size: int,
        d_model: int = 64,
        n_blocks: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        pred_len: int = 1,
        dropout: float = 0.1,
        # v1+ 参数（v0 忽略）
        n_heads: int = 4,
        top_k: int = None,
        patch_sizes: tuple = (4, 8, 16),
        n_snapshots: int = 4,
    ):
        super().__init__()
        self.n_channels  = n_channels
        self.window_size = window_size
        self.d_model     = d_model
        self.pred_len    = pred_len

        # ── 1. 输入嵌入：标量 → D 维 ─────────────────────────────
        self.input_proj = nn.Linear(1, d_model)

        # ── 2. Mamba 编码器（通道间独立，沿 T 轴扫描）────────────
        self.encoder = MambaEncoder(
            d_model=d_model,
            n_blocks=n_blocks,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # ── 3. 预测头：最后时间步 → 预测下一步 ───────────────────
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len),
        )

        # ── 4. 重建头：所有时间步 → 重建输入 ─────────────────────
        self.recon_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, N)

        Returns:
            pred:     (B, N, pred_len)  预测值
            recon:    (B, T, N)         重建值
            score:    (B, N)            异常分数（推理时用）
            aux_loss: scalar(0)         占位，v0 无辅助损失
        """
        B, T, N = x.shape

        # 1. 嵌入：把 N 合并进 batch，对每个通道独立做时序建模
        #    (B, T, N) → (B*N, T, 1) → (B*N, T, D)
        x_in = x.permute(0, 2, 1).reshape(B * N, T, 1)
        h = self.input_proj(x_in)                        # (B*N, T, D)

        # 2. Mamba 编码：(B*N, T, D) → (B*N, T, D)
        h = self.encoder(h)

        # 恢复形状：(B*N, T, D) → (B, N, T, D) → (B, T, N, D)
        h = h.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)

        # 3. 预测：取最后时间步
        last = h[:, -1, :, :]                            # (B, N, D)
        pred = self.pred_head(last)                      # (B, N, pred_len)

        # 4. 重建：所有时间步
        recon = self.recon_head(h).squeeze(-1)           # (B, T, N)

        # 5. 异常分数（推理时用，训练时不用这个）
        pred_err  = (pred.squeeze(-1) - x[:, -1, :]).abs()   # (B, N)
        recon_err = (recon - x).abs().mean(dim=1)             # (B, N)
        score = pred_err + recon_err                          # (B, N)

        aux_loss = x.new_zeros(1).squeeze()

        return pred, recon, score, aux_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        只跑 encoder，返回最后时间步的特征，用于表示空间异常评分。
        供 ablation.py / fast_compare.py 的 repr 评分使用。

        Args:
            x: (B, T, N)
        Returns:
            (B, N, D)
        """
        B, T, N = x.shape
        x_in = x.permute(0, 2, 1).reshape(B * N, T, 1)
        h    = self.input_proj(x_in)                      # (B*N, T, D)
        h    = self.encoder(h)                             # (B*N, T, D)
        last = h[:, -1, :]                                 # (B*N, D)
        return last.reshape(B, N, self.d_model)            # (B, N, D)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 损失函数 v0
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyLoss(nn.Module):
    """
    v0 损失：L = MSE(pred, y) + beta * MSE(recon, x)

    后续版本会在这里叠加 L_freq、L_shape、L_contrast，
    每次只加一项，确保每项都能独立验证效果。

    Args:
        beta: 重建损失权重（默认 0.5）
    """

    def __init__(self, beta: float = 0.5, lambda1: float = 0.0, lambda2: float = 0.0):
        super().__init__()
        self.beta = beta
        self.mse  = nn.MSELoss()
        # lambda1/lambda2 v0 忽略，保留接口兼容性

    def forward(
        self,
        pred:     torch.Tensor,           # (B, N, pred_len)
        target:   torch.Tensor,           # (B, N)
        recon:    torch.Tensor = None,    # (B, T, N)
        x:        torch.Tensor = None,    # (B, T, N)
        aux_loss: torch.Tensor = None,    # scalar，v0 忽略
    ) -> torch.Tensor:

        target_exp = target.unsqueeze(-1).expand_as(pred)
        loss = self.mse(pred, target_exp)

        if recon is not None and x is not None:
            loss = loss + self.beta * self.mse(recon, x)

        return loss


# 兼容旧接口
PredictionLoss = AnomalyLoss
