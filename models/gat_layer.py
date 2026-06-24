"""
动态图注意力网络（修复版）

修复内容：
  - 邻接矩阵：静态节点嵌入 + 动态输入特征相似度（各 50%）
  - 去掉 self-attention 与图结构的粗暴相乘，改为加权融合
  - 标准 GAT 注意力（LeakyReLU + 拼接）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGATLayer(nn.Module):
    """
    动态图注意力层。
    邻接矩阵 = α·静态(节点ID嵌入) + (1-α)·动态(输入特征相似度)
    α 为可学习标量，初始 0.5。

    Args:
        d_model:  节点特征维度
        n_nodes:  节点数
        n_heads:  注意力头数
        dropout:  dropout 概率
        top_k:    每节点保留 top-k 邻居（None=全连接）
    """

    def __init__(self, d_model: int, n_nodes: int, n_heads: int = 4,
                 dropout: float = 0.1, top_k: int = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_nodes  = n_nodes
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.top_k    = top_k

        # ── 静态节点嵌入（图拓扑先验）────────────────────────────
        self.node_emb  = nn.Embedding(n_nodes, d_model)
        self.attn_src  = nn.Linear(d_model, n_heads, bias=False)
        self.attn_dst  = nn.Linear(d_model, n_heads, bias=False)

        # ── 动态图混合权重（可学习，初始化为 0.5）────────────────
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # ── Q / K / V ────────────────────────────────────────────
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

    # ── 静态图（来自节点 ID 嵌入）──────────────────────────────
    def _static_graph(self) -> torch.Tensor:
        """(n_heads, N, N)"""
        ids = torch.arange(self.n_nodes, device=self.node_emb.weight.device)
        emb = self.node_emb(ids)
        src = self.attn_src(emb)   # (N, n_heads)
        dst = self.attn_dst(emb)
        scores = (src.unsqueeze(1) + dst.unsqueeze(0)).permute(2, 0, 1)
        scores = F.leaky_relu(scores, 0.2)
        if self.top_k and self.top_k < self.n_nodes:
            topk_val = scores.topk(self.top_k, dim=-1)[0][..., -1].unsqueeze(-1)
            scores = scores.masked_fill(scores < topk_val, float('-inf'))
        return torch.softmax(scores, dim=-1)  # (n_heads, N, N)

    # ── 动态图（来自输入特征余弦相似度）──────────────────────────
    def _dynamic_graph(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (1, N, N)"""
        x_mean = x.mean(0)                              # (N, D)
        x_norm = F.normalize(x_mean, dim=-1)
        sim = torch.mm(x_norm, x_norm.t())              # (N, N) cosine sim
        sim = torch.softmax(sim / (self.d_model ** 0.5), dim=-1)
        return sim.unsqueeze(0)                          # (1, N, N)

    # ── 前向 ─────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D)"""
        B, N, D = x.shape

        # 1. 混合邻接矩阵
        alpha = torch.sigmoid(self.alpha)
        A = (alpha * self._static_graph()
             + (1 - alpha) * self._dynamic_graph(x))    # (n_heads, N, N)
        A = self.attn_drop(A)

        # 2. 多头 Q/K/V
        def split(t):
            return t.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        Q = split(self.W_q(x)); K = split(self.W_k(x)); V = split(self.W_v(x))

        # 3. 注意力：用图结构软约束（加权平均，不是直接相乘）
        scale = self.d_head ** -0.5
        attn  = torch.matmul(Q, K.transpose(-1, -2)) * scale   # (B,H,N,N)
        # 将图结构作为注意力先验（log-space 加法）
        attn  = attn + torch.log(A.clamp(min=1e-9)).unsqueeze(0)
        attn  = torch.softmax(attn, dim=-1)
        attn  = self.attn_drop(attn)

        # 4. 聚合
        out = torch.matmul(attn, V)                              # (B,H,N,d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.norm(x + self.W_o(out))

    def get_adjacency(self, x: torch.Tensor = None, head_idx: int = 0
                      ) -> torch.Tensor:
        """用于可视化的邻接矩阵"""
        alpha = torch.sigmoid(self.alpha).item()
        A = alpha * self._static_graph()
        if x is not None:
            A = A + (1 - alpha) * self._dynamic_graph(x)
        return A[head_idx].detach().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# 堆叠多层
# ─────────────────────────────────────────────────────────────────────────────

class GraphModule(nn.Module):
    def __init__(self, d_model: int, n_nodes: int, n_heads: int = 4,
                 n_layers: int = 1, dropout: float = 0.1, top_k: int = None):
        super().__init__()
        self.layers = nn.ModuleList([
            DynamicGATLayer(d_model, n_nodes, n_heads, dropout, top_k)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def get_adjacency(self, x: torch.Tensor = None, head_idx: int = 0
                      ) -> torch.Tensor:
        return self.layers[0].get_adjacency(x, head_idx)
