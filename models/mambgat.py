"""
MambGAT-AD — 完整模型（v2，三项改进）

改进 1：多尺度 Patch 嵌入（参考 PSTG）
  - 用 patch_sizes=(4,8,16) 三个尺度切分时序，各自线性投影后门控融合
  - 替换原来的单点 Linear(1, d_model)，让模型同时感知短期波动和长期趋势

改进 2：频域损失（参考 PSTG）
  - L_freq：FFT 频谱 MSE，约束模型刻画正常模式的频率成分
  - L_shape：时序梯度 MSE，约束模型刻画信号形状（上升/下降趋势）
  - 两者使异常时残差更显著，直接提升 AUC

改进 3：图对比正则化（参考 ContrastAD DGCL）
  - 对 batch 内连续时间窗口构建图快照（基于节点特征余弦相似度）
  - 找 KL 散度最大的图对（最不相似的两个时间段）
  - 用 InfoNCE 风格对比损失把它们在潜在空间中推开
  - 使正常/异常点的特征分布在潜在空间中更可分，提升 AUC

整体架构：

  输入 X: (B, T, N)
       │
  [多尺度 Patch 嵌入]   — 3 个尺度 patch → 门控融合 → (B, T', N, D)
       │
  [ST-Mamba-GAT Encoder] — n_blocks 个时空耦合块
       │
  [预测头 + 重建头]      — 预测下一步 + 重建输入窗口
       │
  [异常分数]             — pred_err + recon_err → (B, N)

训练目标：
  L = L_pred + β·L_recon + λ₁·L_freq + λ₂·L_shape + λ_c·L_contrast
"""

from __future__ import annotations

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .st_block import STMambaGATEncoder


# ─────────────────────────────────────────────────────────────────────────────
# 改进 1：多尺度 Patch 嵌入
# ─────────────────────────────────────────────────────────────────────────────

class MultiScalePatchEmbed(nn.Module):
    """
    多尺度时序 Patch 嵌入（参考 PSTG Section 3.2.1）。

    对每个通道，用 K 个不同 patch 大小切分时序，各自线性投影到 D 维，
    再用门控注意力融合为单一表示。

    Args:
        patch_sizes: patch 大小列表，如 (4, 8, 16)
        d_model:     输出嵌入维度
        window_size: 输入时间窗口长度 T
        n_channels:  通道数 N

    输入:  (B, T, N)
    输出:  (B, T_out, N, D)，T_out = T（通过 padding 保持长度）
    """

    def __init__(
        self,
        patch_sizes: Tuple[int, ...],
        d_model: int,
        window_size: int,
        n_channels: int,
    ):
        super().__init__()
        self.patch_sizes  = patch_sizes
        self.d_model      = d_model
        self.window_size  = window_size
        self.n_channels   = n_channels
        self.K            = len(patch_sizes)

        # 每个尺度独立的线性投影（patch_size → d_model）
        self.proj = nn.ModuleList([
            nn.Linear(p, d_model) for p in patch_sizes
        ])

        # 门控融合：K 个嵌入 → 1 个融合嵌入
        # 输入：拼接 K 个 d_model → K*d_model，输出：K 个权重
        self.gate = nn.Linear(self.K * d_model, self.K)

        # 位置编码（可学习）
        self.pos_emb = nn.Parameter(
            torch.randn(1, window_size, 1, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N)
        返回: (B, T, N, D)
        """
        B, T, N = x.shape
        scale_embs = []

        for k, p in enumerate(self.patch_sizes):
            # 用 unfold 切 patch，padding 保持 T 不变
            # unfold: (B, T, N) → 沿 T 轴切 patch
            # 先 pad 左侧 p-1 个 0，使输出长度 = T
            x_pad = F.pad(x, (0, 0, p - 1, 0))  # (B, T+p-1, N)
            # unfold: (B, T, N, p)
            patches = x_pad.unfold(1, p, 1)       # (B, T, N, p)
            # 投影: (B, T, N, p) → (B, T, N, D)
            emb_k = self.proj[k](patches)          # (B, T, N, D)
            scale_embs.append(emb_k)

        # 拼接所有尺度: (B, T, N, K*D)
        concat = torch.cat(scale_embs, dim=-1)     # (B, T, N, K*D)

        # 门控权重: (B, T, N, K)
        weights = torch.softmax(self.gate(concat), dim=-1)

        # 加权融合: sum_k weight_k * emb_k → (B, T, N, D)
        stacked = torch.stack(scale_embs, dim=-2)  # (B, T, N, K, D)
        weights = weights.unsqueeze(-1)             # (B, T, N, K, 1)
        fused   = (stacked * weights).sum(dim=-2)  # (B, T, N, D)

        # 加位置编码
        fused = fused + self.pos_emb[:, :T]

        return fused


# ─────────────────────────────────────────────────────────────────────────────
# 改进 3：图对比正则化（DGCL 简化版）
# ─────────────────────────────────────────────────────────────────────────────

class GraphContrastiveLoss(nn.Module):
    """
    动态图对比正则化（参考 ContrastAD DGCL，Section 3.5）。

    核心思路：
      1. 把 batch 内的时间窗口分成 S 个快照
      2. 每个快照用节点特征的余弦相似度构建图（邻接矩阵）
      3. 找 KL 散度最大的图对 (G_p, G_q)
      4. 用 InfoNCE 风格损失：让 G_p, G_q 各自与"锚图"（其余图均值）对齐，
         同时让 G_p 和 G_q 的嵌入相互分离
      5. 以负权重 λ_c < 0 作为软正则化项加入总损失

    Args:
        d_model:     节点特征维度
        n_snapshots: 把 batch 分成几个快照（默认 4）
        temperature: InfoNCE 温度（默认 0.1）
    """

    def __init__(
        self,
        d_model: int,
        n_snapshots: int = 4,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.n_snapshots = n_snapshots
        self.temperature = temperature
        # GCN 风格的图嵌入投影（简化为单层线性）
        self.graph_proj = nn.Linear(d_model, d_model, bias=False)

    def _build_graph(self, node_feats: torch.Tensor) -> torch.Tensor:
        """
        用节点特征余弦相似度构建归一化邻接矩阵。
        node_feats: (N, D) → 返回 (N, N) 归一化邻接矩阵
        """
        x_norm = F.normalize(node_feats, dim=-1)
        sim = torch.mm(x_norm, x_norm.t())          # (N, N) cosine sim
        # ReLU 去掉负相关，softmax 归一化（行随机矩阵）
        adj = F.softmax(F.relu(sim), dim=-1)
        return adj

    def _graph_embed(
        self, adj: torch.Tensor, node_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        单层 GCN 聚合：z = adj @ node_feats @ W
        adj: (N, N), node_feats: (N, D) → 返回 (D,) 图级嵌入（均值池化）
        """
        h = torch.mm(adj, node_feats)               # (N, D)
        h = self.graph_proj(h)                       # (N, D)
        return h.mean(dim=0)                         # (D,) 图级嵌入

    def _sym_kl(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """对称 KL 散度（用于找最不相似的图对）"""
        a = a.clamp(min=1e-8)
        b = b.clamp(min=1e-8)
        return (a * (a / b).log()).sum() + (b * (b / a).log()).sum()

    def forward(
        self,
        h: torch.Tensor,   # (B, T, N, D) 编码器输出
    ) -> torch.Tensor:
        """
        返回图对比损失（标量）。
        若 batch 太小无法分快照，返回 0。
        """
        B, T, N, D = h.shape
        S = self.n_snapshots

        # 快照长度（沿 T 轴均分）
        snap_len = T // S
        if snap_len < 1:
            return h.new_zeros(1).squeeze()

        # 对 batch 均值（减少随机性），取每个快照的节点特征均值
        h_mean = h.mean(dim=0)  # (T, N, D)

        # 构建 S 个图快照
        adjs   = []
        embeds = []
        for s in range(S):
            t_start = s * snap_len
            t_end   = t_start + snap_len
            snap_feat = h_mean[t_start:t_end].mean(dim=0)  # (N, D)
            adj = self._build_graph(snap_feat)              # (N, N)
            emb = self._graph_embed(adj, snap_feat)         # (D,)
            adjs.append(adj)
            embeds.append(emb)

        embeds = torch.stack(embeds, dim=0)  # (S, D)

        # 找 KL 散度最大的图对 (p, q)
        # 用度分布（行和）近似图的度分布
        best_kl = -1.0
        p_idx, q_idx = 0, 1
        for i in range(S):
            deg_i = adjs[i].sum(dim=-1)  # (N,) 度向量
            deg_i = deg_i / (deg_i.sum() + 1e-8)
            for j in range(i + 1, S):
                deg_j = adjs[j].sum(dim=-1)
                deg_j = deg_j / (deg_j.sum() + 1e-8)
                kl = self._sym_kl(deg_i, deg_j).item()
                if kl > best_kl:
                    best_kl = kl
                    p_idx, q_idx = i, j

        # 锚图嵌入：除 p, q 外其余图的均值
        anchor_mask = [i for i in range(S) if i != p_idx and i != q_idx]
        if len(anchor_mask) == 0:
            # S=2 时退化：用 p+q 均值作锚
            z_anchor = embeds.mean(dim=0, keepdim=True)  # (1, D)
        else:
            z_anchor = embeds[anchor_mask].mean(dim=0, keepdim=True)  # (1, D)

        z_p = embeds[p_idx].unsqueeze(0)  # (1, D)
        z_q = embeds[q_idx].unsqueeze(0)  # (1, D)
        tau = self.temperature

        # InfoNCE 风格：z_p 与 anchor 对齐，z_p 与 z_q 分离
        # score(z_p, z_anchor) / [score(z_p, z_anchor) + score(z_p, z_q)]
        def sim(a, b):
            return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum() / tau

        s_pa = sim(z_p, z_anchor)
        s_qa = sim(z_q, z_anchor)
        s_pq = sim(z_p, z_q)

        # 对比损失（InfoNCE 形式，参考 ContrastAD Eq.17）
        loss_p = -s_pa + torch.log(torch.exp(s_pa) + torch.exp(s_pq) + 1e-8)
        loss_q = -s_qa + torch.log(torch.exp(s_qa) + torch.exp(s_pq) + 1e-8)
        loss   = (loss_p + loss_q) / 2.0

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────────────────────

class MambGATAD(nn.Module):
    """
    MambGAT-AD v2: Spatiotemporal Mamba with Graph Attention
    for Spacecraft Telemetry Anomaly Detection

    Args:
        n_channels:   传感器/通道数（= 图节点数）
        window_size:  输入时间窗口长度 T
        d_model:      特征嵌入维度
        n_blocks:     ST-Mamba-GAT 块数量
        n_heads:      GAT 注意力头数
        d_state:      Mamba SSM 状态维度
        d_conv:       Mamba 局部卷积核大小
        expand:       Mamba 内部维度倍数
        pred_len:     预测步数（默认 1）
        dropout:      dropout 概率
        top_k:        GAT 每节点保留邻居数（None = 全图）
        patch_sizes:  多尺度 patch 大小（默认 (4, 8, 16)）
        n_snapshots:  图对比快照数（默认 4）
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
        patch_sizes: Tuple[int, ...] = (4, 8, 16),
        n_snapshots: int = 4,
    ):
        super().__init__()
        self.n_channels  = n_channels
        self.window_size = window_size
        self.d_model     = d_model
        self.pred_len    = pred_len

        # ── 1. 多尺度 Patch 嵌入（改进 1）──────────────────────────
        self.patch_embed = MultiScalePatchEmbed(
            patch_sizes=patch_sizes,
            d_model=d_model,
            window_size=window_size,
            n_channels=n_channels,
        )

        # ── 2. 时空编码器（核心模块）────────────────────────────────
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

        # ── 3. 预测头（最后一步 → 预测下一步）──────────────────────
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len),
        )

        # ── 4. 重建头（加深，提升重建质量）──────────────────────────
        self.recon_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # ── 5. 图对比正则化模块（改进 3）────────────────────────────
        self.graph_contrast = GraphContrastiveLoss(
            d_model=d_model,
            n_snapshots=n_snapshots,
            temperature=0.1,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, N)  — 滑动窗口输入，N 个通道

        Returns:
            pred:         (B, N, pred_len)  — 预测值
            recon:        (B, T, N)         — 重建值
            score:        (B, N)            — 异常分数
            contrast_loss: scalar           — 图对比损失（训练时用）
        """
        B, T, N = x.shape

        # 1. 多尺度 Patch 嵌入：(B, T, N) → (B, T, N, D)
        h = self.patch_embed(x)

        # 2. 时空编码：(B, T, N, D) → (B, T, N, D)
        h = self.encoder(h)

        # 3. 预测：取最后一个时间步
        last = h[:, -1, :, :]                          # (B, N, D)
        pred = self.pred_head(last)                     # (B, N, pred_len)

        # 4. 重建：所有时间步还原输入
        recon = self.recon_head(h).squeeze(-1)          # (B, T, N)

        # 5. 图对比损失（改进 3）
        contrast_loss = self.graph_contrast(h)

        # 6. 联合异常分数
        pred_err  = (pred.squeeze(-1) - x[:, -1, :]).abs()    # (B, N)
        recon_err = (recon - x).abs().mean(dim=1)              # (B, N)
        score = pred_err + recon_err                           # (B, N)

        return pred, recon, score, contrast_loss

    # ─────────────────────────────────────────────────────────────────
    def get_graph(self, head_idx: int = 0) -> torch.Tensor:
        """获取学习到的传感器耦合图（用于论文可视化）。返回 (N, N) 邻接矩阵"""
        return self.encoder.blocks[0].spatial.get_adjacency(head_idx=head_idx)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 改进 2：频域损失（参考 PSTG Eq.25）
# ─────────────────────────────────────────────────────────────────────────────

class PredictionLoss(nn.Module):
    """
    联合损失（v2）：
      L = L_pred + β·L_recon + λ₁·L_freq + λ₂·L_shape + λ_c·L_contrast

    L_pred:    预测损失（MSE + MAE 混合）
    L_recon:   重建损失（MSE）
    L_freq:    频域损失（FFT 频谱 MSE，参考 PSTG）
    L_shape:   形状损失（时序梯度 MSE，参考 PSTG）
    L_contrast: 图对比正则化（参考 ContrastAD DGCL）

    Args:
        alpha:    pred_loss 中 MAE 权重（0=纯MSE, 1=纯MAE）
        beta:     重建损失权重
        lambda1:  频域损失权重
        lambda2:  形状损失权重
        lambda_c: 图对比损失权重（负值=软正则化，推开异常/正常分布）
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.1,
        lambda1: float = 0.1,
        lambda2: float = 0.05,
        lambda_c: float = -0.4,
    ):
        super().__init__()
        self.alpha    = alpha
        self.beta     = beta
        self.lambda1  = lambda1
        self.lambda2  = lambda2
        self.lambda_c = lambda_c
        self.mse      = nn.MSELoss()
        self.mae      = nn.L1Loss()

    def forward(
        self,
        pred:          torch.Tensor,          # (B, N, pred_len)
        target:        torch.Tensor,          # (B, N)
        recon:         torch.Tensor = None,   # (B, T, N)
        x:             torch.Tensor = None,   # (B, T, N)
        contrast_loss: torch.Tensor = None,   # scalar
    ) -> torch.Tensor:

        # ── 预测损失 ──────────────────────────────────────────────
        target_exp = target.unsqueeze(-1).expand_as(pred)
        pred_loss  = (1 - self.alpha) * self.mse(pred, target_exp) \
                   + self.alpha       * self.mae(pred, target_exp)

        total = pred_loss

        # ── 重建损失 ──────────────────────────────────────────────
        if recon is not None and x is not None:
            recon_loss = self.mse(recon, x)
            total = total + self.beta * recon_loss

            # ── 频域损失（改进 2）─────────────────────────────────
            # FFT 沿时间轴，取幅度谱
            fft_x     = torch.fft.rfft(x,     dim=1, norm="ortho")
            fft_recon = torch.fft.rfft(recon, dim=1, norm="ortho")
            freq_loss = self.mse(fft_recon.abs(), fft_x.abs())
            total = total + self.lambda1 * freq_loss

            # ── 形状损失（时序梯度 MSE）──────────────────────────
            # 用 diff 近似时序梯度
            grad_x     = x[:, 1:, :]     - x[:, :-1, :]      # (B, T-1, N)
            grad_recon = recon[:, 1:, :] - recon[:, :-1, :]
            shape_loss = self.mse(grad_recon, grad_x)
            total = total + self.lambda2 * shape_loss

        # ── 图对比正则化（改进 3）────────────────────────────────
        if contrast_loss is not None:
            total = total + self.lambda_c * contrast_loss

        return total
