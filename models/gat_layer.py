"""
动态图注意力网络（GAT）— 可学习邻接矩阵，无需先验图结构

核心设计：
  • 可学习邻接矩阵（Learnable Adjacency Matrix）：
    从数据中动态发现传感器间的耦合关系，无需手工标注拓扑
  • 多头注意力（Multi-Head Attention）：
    并行捕捉不同类型的传感器关系（热耦合、电耦合等）
  • 无需 PyTorch Geometric：纯矩阵操作，Windows 原生可用

参考：
  Veličković et al., "Graph Attention Networks", ICLR 2018
  Deng & Hooi, "Graph Deviation Network for Cloud Service Anomaly
  Detection", AAAI 2021  (GDN)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGATLayer(nn.Module):
    """
    单层动态图注意力，可学习邻接矩阵。

    Args:
        d_model:    节点特征维度
        n_nodes:    节点数（= 传感器/通道数）
        n_heads:    注意力头数
        dropout:    attention dropout 概率
        top_k:      稀疏化：每个节点只保留 top-k 个邻居（None = 全图）
    """

    def __init__(
        self,
        d_model: int,
        n_nodes: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        top_k: int = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须整除 n_heads"
        self.d_model = d_model
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.top_k   = top_k

        # ── 可学习邻接矩阵（节点嵌入用于计算图结构）────────────────
        # 每个节点学习一个 embedding，节点间相似度决定图结构
        # 参考 GDN 的做法，但改为软注意力权重
        self.node_emb = nn.Embedding(n_nodes, d_model)

        # ── Q / K / V 线性投影（多头）─────────────────────────────
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # ── 图结构注意力（基于可学习节点嵌入）─────────────────────
        self.attn_src = nn.Linear(d_model, n_heads, bias=False)
        self.attn_dst = nn.Linear(d_model, n_heads, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # ── 节点特征投影后的归一化 ──────────────────────────────
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

    # ─────────────────────────────────────────────────────────────────
    def _compute_graph(self) -> torch.Tensor:
        """
        基于可学习节点嵌入计算稀疏注意力图。
        返回归一化的邻接权重矩阵 A: (n_heads, N, N)
        """
        node_ids = torch.arange(self.n_nodes, device=self.node_emb.weight.device)
        emb = self.node_emb(node_ids)  # (N, d_model)

        # 对每个 head 分别计算 src-dst 注意力得分
        src = self.attn_src(emb)  # (N, n_heads)
        dst = self.attn_dst(emb)  # (N, n_heads)

        # 外加得分：A_ij = e_i + e_j  — 线性复杂度
        scores = src.unsqueeze(1) + dst.unsqueeze(0)  # (N, N, n_heads)
        scores = scores.permute(2, 0, 1)              # (n_heads, N, N)
        scores = F.leaky_relu(scores, negative_slope=0.2)

        # 可选稀疏化：保留 top-k 邻居
        if self.top_k is not None and self.top_k < self.n_nodes:
            topk_val, _ = scores.topk(self.top_k, dim=-1)
            threshold = topk_val[..., -1].unsqueeze(-1)
            scores = scores.masked_fill(scores < threshold, float('-inf'))

        # Softmax 归一化
        A = torch.softmax(scores, dim=-1)             # (n_heads, N, N)
        return self.attn_drop(A)

    # ─────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)  — 每个时间步的节点特征
        Returns:
            out: (B, N, d_model)
        """
        B, N, D = x.shape

        # 1. 计算图结构
        A = self._compute_graph()    # (n_heads, N, N)

        # 2. 多头 Q / K / V
        def split_heads(t):
            # t: (B, N, D) → (B, n_heads, N, d_head)
            return t.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        Q = split_heads(self.W_q(x))   # (B, H, N, d_head)
        K = split_heads(self.W_k(x))
        V = split_heads(self.W_v(x))

        # 3. 注意力加权聚合（图结构引导）
        #    self-attention 得分
        scale = self.d_head ** -0.5
        attn = torch.matmul(Q, K.transpose(-1, -2)) * scale   # (B, H, N, N)

        # 与图结构融合（element-wise 乘积）
        attn = attn * A.unsqueeze(0)   # broadcast batch
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 4. 聚合 V
        out = torch.matmul(attn, V)    # (B, H, N, d_head)
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        out = self.W_o(out)

        # 5. 残差 + 归一化
        return self.norm(x + out)


# ─────────────────────────────────────────────────────────────────────────────
# 完整 GAT 模块（可堆叠多层）
# ─────────────────────────────────────────────────────────────────────────────

class GraphModule(nn.Module):
    """
    堆叠 n_layers 个 DynamicGATLayer，提取空间依赖特征。

    同时输出学习到的邻接矩阵，用于论文可视化（物理耦合关系图）。
    """

    def __init__(self, d_model: int, n_nodes: int, n_heads: int = 4,
                 n_layers: int = 1, dropout: float = 0.1, top_k: int = None):
        super().__init__()
        self.layers = nn.ModuleList([
            DynamicGATLayer(d_model, n_nodes, n_heads, dropout, top_k)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model) → (B, N, d_model)"""
        for layer in self.layers:
            x = layer(x)
        return x

    def get_adjacency(self, head_idx: int = 0) -> torch.Tensor:
        """
        获取第一层第 head_idx 个注意力头的邻接矩阵（用于可视化）。
        返回 (N, N) 归一化权重矩阵
        """
        return self.layers[0]._compute_graph()[head_idx].detach().cpu()
