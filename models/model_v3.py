"""
MambGAT-AD v3 — 在 v1 基础上加入多尺度 Patch 嵌入

v1 → v3 的唯一改动：输入嵌入层
  v1: Linear(1, D)  — 每步一个标量直接投影
  v3: MultiScalePatchEmbed(patch_sizes=(4,8,16), D)
      — 3 个尺度切 patch，各自线性投影后门控融合

动机（参考 PSTG / CATCH）：
  单点嵌入只看当前时刻，patch 嵌入同时感知短期波动（小patch）
  和长期趋势（大patch），让 Mamba 在更丰富的上下文里建模时序。

模型结构：
  输入 X: (B, T, N)
       │
  [MultiScalePatchEmbed]   3 个尺度 → 门控融合  →  (B, T, N, D)
       │  reshape → (B*N, T, D)
  [Mamba Encoder]          通道独立时序建模      →  (B*N, T, D)
       │  reshape → (B, T, N, D)
  [Dynamic GAT]            跨通道空间聚合        →  (B, T, N, D)
       │
  [Pred Head]              最后时间步 → 预测     →  (B, N, 1)
  [Recon Head]             所有时间步 → 重建     →  (B, T, N)

损失（与 v1 相同，不加频域损失）：
  L = MSE(pred, y) + β·MSE(recon, x)

ablation 目的：
  对比 v1 vs v3，量化多尺度 Patch 嵌入对 AUC 的贡献。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_v0 import MambaEncoder       # Mamba 编码器，复用 v0
from .model_v1 import DynamicGATLayer    # 动态 GAT，复用 v1


# ─────────────────────────────────────────────────────────────────────────────
# 多尺度 Patch 嵌入（v3 核心新增，参考 PSTG Section 3.2.1）
# ─────────────────────────────────────────────────────────────────────────────

class MultiScalePatchEmbed(nn.Module):
    """
    多尺度时序 Patch 嵌入。

    对 N 个通道，用 K 个不同 patch 大小切分时序，各自线性投影到 D 维，
    再用门控注意力融合为单一表示。

    Args:
        patch_sizes: patch 大小列表，如 (4, 8, 16)
        d_model:     输出嵌入维度 D
        window_size: 输入时间窗口长度 T

    输入:  (B, T, N)
    输出:  (B, T, N, D)
    """

    def __init__(
        self,
        patch_sizes: Tuple[int, ...],
        d_model: int,
        window_size: int,
    ):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.K = len(patch_sizes)

        # 每个尺度独立的线性投影 (patch_size → d_model)
        self.proj = nn.ModuleList([
            nn.Linear(p, d_model) for p in patch_sizes
        ])

        # 门控融合：拼接 K 个 d_model → K 个权重
        self.gate = nn.Linear(self.K * d_model, self.K)

        # 可学习位置编码
        self.pos_emb = nn.Parameter(
            torch.randn(1, window_size, 1, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N) → (B, T, N, D)"""
        B, T, N = x.shape
        scale_embs = []

        for k, p in enumerate(self.patch_sizes):
            # 左 padding 保持长度 T 不变
            x_pad = F.pad(x, (0, 0, p - 1, 0))     # (B, T+p-1, N)
            patches = x_pad.unfold(1, p, 1)          # (B, T, N, p)
            emb_k = self.proj[k](patches)             # (B, T, N, D)
            scale_embs.append(emb_k)

        # 门控融合
        concat  = torch.cat(scale_embs, dim=-1)      # (B, T, N, K*D)
        weights = torch.softmax(self.gate(concat), dim=-1).unsqueeze(-1)
                                                      # (B, T, N, K, 1)
        stacked = torch.stack(scale_embs, dim=-2)    # (B, T, N, K, D)
        fused   = (stacked * weights).sum(dim=-2)    # (B, T, N, D)

        return fused + self.pos_emb[:, :T]


# ─────────────────────────────────────────────────────────────────────────────
# 主模型 v3
# ─────────────────────────────────────────────────────────────────────────────

class MambGATAD(nn.Module):
    """
    MambGAT-AD v3：MultiScalePatchEmbed + Mamba + Dynamic GAT + Pred/Recon

    v1 vs v3：只有输入嵌入层不同。
    """

    def __init__(
        self,
        n_channels:  int,
        window_size: int,
        d_model:     int   = 64,
        n_blocks:    int   = 2,
        d_state:     int   = 16,
        d_conv:      int   = 4,
        expand:      int   = 2,
        pred_len:    int   = 1,
        dropout:     float = 0.1,
        n_heads:     int   = 4,
        # v3 参数
        patch_sizes: tuple = (4, 8, 16),
        # 以下忽略，保留接口兼容性
        top_k:       int   = None,
        n_snapshots: int   = 4,
    ):
        super().__init__()
        self.n_channels  = n_channels
        self.window_size = window_size
        self.d_model     = d_model
        self.pred_len    = pred_len

        # ── 1. 多尺度 Patch 嵌入（v3 替换 v1 的 Linear(1, D)）────
        self.patch_embed = MultiScalePatchEmbed(
            patch_sizes=patch_sizes,
            d_model=d_model,
            window_size=window_size,
        )

        # ── 2. Mamba 编码器（与 v0/v1 相同）──────────────────────
        self.encoder = MambaEncoder(
            d_model=d_model, n_blocks=n_blocks,
            d_state=d_state, d_conv=d_conv,
            expand=expand, dropout=dropout,
        )

        # ── 3. Dynamic GAT（与 v1 相同）──────────────────────────
        self.gat      = DynamicGATLayer(d_model, n_channels, n_heads, dropout)
        self.gat_gate = nn.Parameter(torch.zeros(1))

        # ── 4. 预测头（与 v0/v1 相同）────────────────────────────
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len),
        )

        # ── 5. 重建头（与 v0/v1 相同）────────────────────────────
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, T, N = x.shape

        # 1. 多尺度 Patch 嵌入：(B, T, N) → (B, T, N, D)
        h = self.patch_embed(x)

        # 2. Mamba：reshape 为 (B*N, T, D) 后通道独立处理
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        h = self.encoder(h)
        h = h.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)  # (B,T,N,D)

        # 3. Dynamic GAT：跨通道空间聚合
        h_flat = h.reshape(B * T, N, self.d_model)
        h_gat  = self.gat(h_flat).reshape(B, T, N, self.d_model)
        h = h + torch.sigmoid(self.gat_gate) * h_gat

        # 4. 预测头
        pred  = self.pred_head(h[:, -1, :, :])        # (B, N, pred_len)

        # 5. 重建头
        recon = self.recon_head(h).squeeze(-1)         # (B, T, N)

        # 6. 异常分数（推理占位）
        pred_err  = (pred.squeeze(-1) - x[:, -1, :]).abs()
        recon_err = (recon - x).abs().mean(dim=1)
        score     = pred_err + recon_err

        return pred, recon, score, x.new_zeros(1).squeeze()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        h = self.patch_embed(x)
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        h = self.encoder(h)
        h = h.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)
        h_flat = h.reshape(B * T, N, self.d_model)
        h_gat  = self.gat(h_flat).reshape(B, T, N, self.d_model)
        h = h + torch.sigmoid(self.gat_gate) * h_gat
        return h[:, -1, :, :]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# AnomalyLoss 与 v1 相同（不加频域损失）
from .model_v0 import AnomalyLoss  # noqa: F401, E402
