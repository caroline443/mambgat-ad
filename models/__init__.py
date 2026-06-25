"""
MambGAT-AD 模型包

版本历史（渐进式叠加，每版独立可验证）：
  v0  LinearEmbed + Mamba Encoder + 预测头
  v1  + 动态 GAT（空间建模）                        ← 当前
  v2  + 频域损失（L_freq + L_shape）
  v3  + 多尺度 Patch 嵌入
  v4  + 图对比正则化（DGCL）
"""

from .model_v1 import MambGATAD
from .model_v0 import AnomalyLoss   # 损失函数 v0/v1 共用

__all__ = ["MambGATAD", "AnomalyLoss"]
