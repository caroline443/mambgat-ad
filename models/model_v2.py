"""
MambGAT-AD v2 — 在 v1 基础上加入频域损失

v1 → v2 的唯一改动：损失函数新增两项
  L_freq:  FFT 幅度谱 MSE  — 约束模型在频率维度上刻画正常模式
  L_shape: 时序梯度 MSE    — 约束模型保留信号的上升/下降形状

模型结构与 v1 完全相同（Linear Embed → Mamba → Dynamic GAT → Pred/Recon Head），
只有 AnomalyLoss 不同，确保 ablation 对比可信。

ablation 目的：
  对比 v1 vs v2，量化频域损失对 AUC 的贡献。
"""

from __future__ import annotations

import torch
import torch.nn as nn

# 模型结构直接复用 v1（无任何改动）
from .model_v1 import MambGATAD  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# 损失函数 v2（在 v1 基础上加频域损失）
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyLoss(nn.Module):
    """
    v2 损失：L = MSE(pred, y) + β·MSE(recon, x) + λ₁·L_freq + λ₂·L_shape

    L_freq:  FFT 幅度谱 MSE
      fft_x     = |RFFT(x, dim=T)|     正常模式的频率成分
      fft_recon = |RFFT(recon, dim=T)| 重建的频率成分
      L_freq    = MSE(fft_recon, fft_x)
      → 强迫模型在频谱上也忠实地重建正常模式，
        异常时频谱偏差更大，anomaly score 更显著

    L_shape: 时序梯度 MSE
      grad_x     = x[:,1:,:] - x[:,:-1,:]       原始信号的一阶差分
      grad_recon = recon[:,1:,:] - recon[:,:-1,:]
      L_shape    = MSE(grad_recon, grad_x)
      → 约束模型保留信号的局部变化趋势（上升/下降形状），
        对形状异常更敏感

    Args:
        beta:    重建损失权重（默认 0.5，与 v0/v1 一致）
        lambda1: 频域损失权重
        lambda2: 形状损失权重
    """

    def __init__(
        self,
        beta:    float = 0.5,
        lambda1: float = 0.1,
        lambda2: float = 0.05,
    ):
        super().__init__()
        self.beta    = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mse     = nn.MSELoss()

    def forward(
        self,
        pred:     torch.Tensor,           # (B, N, pred_len)
        target:   torch.Tensor,           # (B, N)
        recon:    torch.Tensor = None,    # (B, T, N)
        x:        torch.Tensor = None,    # (B, T, N)
        aux_loss: torch.Tensor = None,    # scalar，v2 忽略
    ) -> torch.Tensor:

        # ── 预测损失（与 v0/v1 相同）──────────────────────────────
        target_exp = target.unsqueeze(-1).expand_as(pred)
        loss = self.mse(pred, target_exp)

        if recon is not None and x is not None:

            # ── 重建损失（与 v0/v1 相同）──────────────────────────
            loss = loss + self.beta * self.mse(recon, x)

            # ── 频域损失（v2 新增）────────────────────────────────
            # RFFT 沿时间轴，取幅度谱（忽略相位）
            fft_x     = torch.fft.rfft(x,     dim=1, norm="ortho").abs()
            fft_recon = torch.fft.rfft(recon, dim=1, norm="ortho").abs()
            loss = loss + self.lambda1 * self.mse(fft_recon, fft_x)

            # ── 形状损失（v2 新增）────────────────────────────────
            # 一阶差分近似时序梯度
            grad_x     = x[:, 1:, :]     - x[:, :-1, :]
            grad_recon = recon[:, 1:, :] - recon[:, :-1, :]
            loss = loss + self.lambda2 * self.mse(grad_recon, grad_x)

        return loss
