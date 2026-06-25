"""
MambGAT-AD v1 — 在 v0 基础上加入动态 GAT（空间建模）

v0 → v1 的唯一改动：
  在 Mamba Encoder 之后、Pred/Recon Head 之前，
  插入一个 Dynamic GAT 层，让通道间可以互相感知。

架构：

  输入 X: (B, T, N)
       │
  [Linear Embed]      每个通道独立投影到 D 维  →  (B*N, T, D)
       │
  [Mamba Encoder]     沿 T 轴扫描（通道间仍独立）
  (B*N, T, D)
       │
  reshape → (B, T, N, D)
       │
  [Dynamic GAT]       ← v1 新增：让通道间互相聚合信息
  每个时间步对 N 个节点做图注意力
  (B, T, N, D) → flatten T → (B*T, N, D) → GAT → (B, T, N, D)
       │
  [Pred Head]         取最后时间步 → 预测下一步  →  (B, N, 1)
       │
  [Recon Head]        所有时间步 → 重建输入      →  (B, T, N)

动态 GAT 邻接矩阵：
  A = σ(α) · A_static + (1 - σ(α)) · A_dynamic
  A_static:  从可学习节点嵌入计算（捕捉稳定的传感器耦合拓扑）
  A_dynamic: 从当前输入特征余弦相似度计算（捕捉运行时变化）
  α:         可学习标量，控制静/动态图的混合比例

损失（与 v0 相同）：
  L = MSE(pred, y) + beta * MSE(recon, x)

ablation 目的：
  对比 v0 vs v1，量化动态 GAT 对 AUC 的贡献。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_v0 import MambaEncoder   # 直接复用 v0 的 Mamba 编码器


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic GAT（v1 核心新增）
# ─────────────────────────────────────────────────────────────────────────────

class DynamicGATLayer(nn.Module):
    """
    动态图注意力层。

    邻接矩阵 = σ(α) · A_static + (1-σ(α)) · A_dynamic
      A_static:  可学习节点嵌入 → LeakyReLU 注意力 → softmax
      A_dynamic: 当前特征余弦相似度 → softmax

    对 N 个节点做多头注意力，以混合邻接矩阵为先验。

    Args:
        d_model:  节点特征维度
        n_nodes:  节点数（传感器数）
        n_heads:  注意力头数
        dropout:  dropout 概率
    """

    def __init__(self, d_model: int, n_nodes: int,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.n_nodes  = n_nodes

        # ── 静态图（节点 ID 嵌入）─────────────────────────────
        self.node_emb  = nn.Embedding(n_nodes, d_model)
        self.attn_src  = nn.Linear(d_model, n_heads, bias=False)
        self.attn_dst  = nn.Linear(d_model, n_heads, bias=False)

        # ── 混合权重（初始 0.5，可学习）──────────────────────
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # ── 多头注意力 QKV ────────────────────────────────────
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

    def _static_adj(self) -> torch.Tensor:
        """(n_heads, N, N)  从节点嵌入计算静态邻接矩阵"""
        ids = torch.arange(self.n_nodes,
                           device=self.node_emb.weight.device)
        e   = self.node_emb(ids)                        # (N, D)
        src = self.attn_src(e)                           # (N, n_heads)
        dst = self.attn_dst(e)                           # (N, n_heads)
        # (n_heads, N, N)
        scores = (src.unsqueeze(1) + dst.unsqueeze(0)).permute(2, 0, 1)
        scores = F.leaky_relu(scores, 0.2)
        return torch.softmax(scores, dim=-1)

    def _dynamic_adj(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (1, N, N)  从特征余弦相似度计算动态邻接矩阵"""
        x_mean = x.mean(dim=0)                           # (N, D)
        x_norm = F.normalize(x_mean, dim=-1)
        sim = torch.mm(x_norm, x_norm.t())               # (N, N)
        sim = torch.softmax(sim / (self.d_head ** 0.5), dim=-1)
        return sim.unsqueeze(0)                           # (1, N, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        返回: (B, N, D)
        """
        B, N, D = x.shape

        # 混合邻接矩阵 (n_heads, N, N)
        alpha = torch.sigmoid(self.alpha)
        A = (alpha * self._static_adj()
             + (1 - alpha) * self._dynamic_adj(x))       # (n_heads, N, N)
        A = self.drop(A)

        # 多头 QKV
        def split_heads(t):
            return t.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
                                                          # (B, H, N, d_head)
        Q = split_heads(self.W_q(x))
        K = split_heads(self.W_k(x))
        V = split_heads(self.W_v(x))

        # 注意力分数 + 图结构先验（log-space）
        scale = self.d_head ** -0.5
        attn  = torch.matmul(Q, K.transpose(-1, -2)) * scale  # (B, H, N, N)
        attn  = attn + torch.log(A.clamp(min=1e-9)).unsqueeze(0)
        attn  = torch.softmax(attn, dim=-1)
        attn  = self.drop(attn)

        # 聚合
        out = torch.matmul(attn, V)                            # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.norm(x + self.W_o(out))


# ─────────────────────────────────────────────────────────────────────────────
# 主模型 v1
# ─────────────────────────────────────────────────────────────────────────────

class MambGATAD(nn.Module):
    """
    MambGAT-AD v1：Mamba Encoder + Dynamic GAT + Pred/Recon Head

    v0 vs v1 的唯一区别：加了 DynamicGATLayer。
    其余（嵌入、头、损失、评估）完全相同，确保对比可信。
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
        # GAT 参数（v1 新增）
        n_heads:     int   = 4,
        # 以下参数 v1 忽略，保留接口兼容性
        top_k:        int   = None,
        patch_sizes:  tuple = (4, 8, 16),
        n_snapshots:  int   = 4,
    ):
        super().__init__()
        self.n_channels  = n_channels
        self.window_size = window_size
        self.d_model     = d_model
        self.pred_len    = pred_len

        # ── 1. 输入嵌入（与 v0 相同）──────────────────────────
        self.input_proj = nn.Linear(1, d_model)

        # ── 2. Mamba 编码器（与 v0 相同）──────────────────────
        self.encoder = MambaEncoder(
            d_model=d_model, n_blocks=n_blocks,
            d_state=d_state, d_conv=d_conv,
            expand=expand, dropout=dropout,
        )

        # ── 3. Dynamic GAT（v1 新增）──────────────────────────
        self.gat = DynamicGATLayer(
            d_model=d_model,
            n_nodes=n_channels,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.gat_gate = nn.Parameter(torch.zeros(1))  # 初始化为 0，让 GAT 平滑加入

        # ── 4. 预测头（与 v0 相同）────────────────────────────
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len),
        )

        # ── 5. 重建头（与 v0 相同）────────────────────────────
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
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T, N)
        返回: pred (B,N,pred_len), recon (B,T,N), score (B,N), aux_loss=0
        """
        B, T, N = x.shape

        # 1. 嵌入：每通道独立
        x_in = x.permute(0, 2, 1).reshape(B * N, T, 1)
        h    = self.input_proj(x_in)           # (B*N, T, D)

        # 2. Mamba：每通道独立时序建模
        h = self.encoder(h)                    # (B*N, T, D)

        # reshape → (B, T, N, D)
        h = h.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)

        # 3. Dynamic GAT：跨通道空间聚合（v1 新增）
        # 对所有时间步并行做 GAT：(B, T, N, D) → (B*T, N, D) → GAT → (B, T, N, D)
        h_flat = h.reshape(B * T, N, self.d_model)
        h_gat  = self.gat(h_flat)              # (B*T, N, D)
        h_gat  = h_gat.reshape(B, T, N, self.d_model)

        # 残差门控：初始训练时 GAT 权重接近 0，平滑融入
        h = h + torch.sigmoid(self.gat_gate) * h_gat

        # 4. 预测头
        last = h[:, -1, :, :]                  # (B, N, D)
        pred = self.pred_head(last)             # (B, N, pred_len)

        # 5. 重建头
        recon = self.recon_head(h).squeeze(-1) # (B, T, N)

        # 6. 异常分数（推理时用）
        pred_err  = (pred.squeeze(-1) - x[:, -1, :]).abs()
        recon_err = (recon - x).abs().mean(dim=1)
        score     = pred_err + recon_err

        aux_loss = x.new_zeros(1).squeeze()
        return pred, recon, score, aux_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """供 ablation.py repr 评分使用：返回 GAT 后最后时间步特征 (B, N, D)"""
        B, T, N = x.shape
        x_in = x.permute(0, 2, 1).reshape(B * N, T, 1)
        h    = self.input_proj(x_in)
        h    = self.encoder(h)
        h    = h.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)
        h_flat = h.reshape(B * T, N, self.d_model)
        h_gat  = self.gat(h_flat).reshape(B, T, N, self.d_model)
        h = h + torch.sigmoid(self.gat_gate) * h_gat
        return h[:, -1, :, :]                  # (B, N, D)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 兼容旧接口
AnomalyLoss = None   # 继续用 model_v0 的 AnomalyLoss
