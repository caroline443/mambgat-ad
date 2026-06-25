"""
快速验证 v2 模型的 forward / loss 是否正常。
"""
import sys, torch
sys.path.insert(0, ".")
from models.mambgat import MambGATAD, PredictionLoss

B, T, N = 4, 100, 25   # SMAP: 25 channels, window=100

model = MambGATAD(
    n_channels  = N,
    window_size = T,
    d_model     = 64,
    n_blocks    = 2,
    n_heads     = 4,
    d_state     = 16,
    d_conv      = 4,
    expand      = 2,
    pred_len    = 1,
    dropout     = 0.1,
    patch_sizes = (4, 8, 16),
    n_snapshots = 4,
)
print(f"参数量: {model.count_parameters():,}")

x      = torch.randn(B, T, N)
target = torch.randn(B, N)

pred, recon, score, contrast_loss = model(x)
print(f"pred:          {tuple(pred.shape)}")
print(f"recon:         {tuple(recon.shape)}")
print(f"score:         {tuple(score.shape)}")
print(f"contrast_loss: {contrast_loss.item():.6f}")

criterion = PredictionLoss(alpha=0.5, beta=0.1, lambda1=0.1, lambda2=0.05, lambda_c=-0.4)
loss = criterion(pred, target, recon=recon, x=x, contrast_loss=contrast_loss)
print(f"total loss:    {loss.item():.6f}")

# 反向传播验证梯度
loss.backward()
print("backward OK — 所有梯度正常")

# 验证 score 分布（不应全为 0 或 nan）
import numpy as np
s = score.detach().numpy()
print(f"score stats: min={s.min():.4f}  max={s.max():.4f}  mean={s.mean():.4f}  nan={np.isnan(s).any()}")
