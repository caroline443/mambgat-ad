"""
诊断脚本：逐步检查数据加载 → DataLoader → 模型前向传播
找出 loss=nan / DataLoader 空的根本原因

用法：
  python debug.py --config config/smap_win.yaml
"""

import argparse
import numpy as np
import torch
import yaml

# ── 1. 加载配置 ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/smap_win.yaml")
args = parser.parse_args()
with open(args.config, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

data_dir   = cfg["data"]["data_dir"]
label_file = cfg["data"]["label_file"]
dataset    = cfg["data"]["dataset"]
window_size = cfg["data"]["window_size"]
batch_size  = cfg["train"]["batch_size"]

print("=" * 60)
print("  MambGAT-AD 诊断脚本")
print("=" * 60)

# ── 2. 直接加载通道，检查数据质量 ─────────────────────────────────────────────
print("\n[步骤1] 检查原始数据...")
from data.dataset import get_channels, _load_channel

channels = get_channels(label_file, dataset)
print(f"  CSV 通道数: {len(channels)}")

# 加载每个通道，打印长度和 NaN/Inf 统计
train_lens = []
bad_channels = []
for ch in channels:
    try:
        s = _load_channel(data_dir, "train", ch)
        has_nan = not np.isfinite(s).all()
        train_lens.append((ch, len(s), has_nan))
        if has_nan:
            bad_channels.append(ch)
    except Exception as e:
        print(f"  [ERROR] {ch}: {e}")

train_lens.sort(key=lambda x: x[1])
print(f"\n  最短 5 个通道（决定 min_len）:")
for ch, l, bad in train_lens[:5]:
    print(f"    {ch:8s}  len={l:6d}  {'⚠️ 含 NaN/Inf' if bad else 'ok'}")
print(f"\n  最长 5 个通道:")
for ch, l, bad in train_lens[-5:]:
    print(f"    {ch:8s}  len={l:6d}  {'⚠️ 含 NaN/Inf' if bad else 'ok'}")

print(f"\n  含 NaN/Inf 的通道数: {len(bad_channels)}")
if bad_channels:
    print(f"  通道列表: {bad_channels[:10]}")

min_len = train_lens[0][1]
print(f"\n  min_len = {min_len}  (window_size={window_size})")
if min_len <= window_size:
    print(f"  [CRITICAL] min_len={min_len} <= window_size={window_size}，数据集将为空！")
else:
    expected_train = (min_len - window_size) // 5
    print(f"  预期训练样本数 ≈ {expected_train}  (step=5)")

# ── 3. 检查 DataLoader ────────────────────────────────────────────────────────
print("\n[步骤2] 构建 DataLoader...")
from data import build_loaders

train_loader, test_loader, test_labels, n_channels = build_loaders(
    data_dir=data_dir, label_file=label_file, dataset=dataset,
    window_size=window_size, train_step=5, test_step=1,
    batch_size=batch_size, normalize_data=True, num_workers=0,
)

print(f"\n  DataLoader 报告批次数: {len(train_loader)}")
print(f"  (应 = {len(train_loader.dataset)} // {batch_size} = "
      f"{len(train_loader.dataset) // batch_size})")

# 尝试取第一批
print("\n[步骤3] 取第一批数据...")
try:
    x, y = next(iter(train_loader))
    print(f"  x.shape = {x.shape}  dtype={x.dtype}")
    print(f"  y.shape = {y.shape}  dtype={y.dtype}")
    print(f"  x 含 NaN? {torch.isnan(x).any().item()}")
    print(f"  x 含 Inf? {torch.isinf(x).any().item()}")
    print(f"  x 范围: [{x.min():.4f}, {x.max():.4f}]")
except StopIteration:
    print("  [CRITICAL] DataLoader 为空，无法取到任何批次！")
    print("  → 检查上面的 min_len 是否 <= window_size")
    raise SystemExit(1)
except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {e}")
    raise

# ── 4. 模型前向传播 ───────────────────────────────────────────────────────────
print("\n[步骤4] 模型前向传播...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  设备: {device}")

from models import MambGATAD, PredictionLoss

model = MambGATAD(
    n_channels=n_channels, window_size=window_size,
    d_model=cfg["model"]["d_model"], n_blocks=cfg["model"]["n_blocks"],
    n_heads=cfg["model"]["n_heads"], d_state=cfg["model"]["d_state"],
    d_conv=cfg["model"]["d_conv"], expand=cfg["model"]["expand"],
    pred_len=cfg["model"]["pred_len"], dropout=0.0,
).to(device)

x = x.to(device, dtype=torch.float32)
y = y.to(device, dtype=torch.float32)

with torch.no_grad():
    pred, score = model(x)

print(f"  pred.shape  = {pred.shape}")
print(f"  pred 含 NaN? {torch.isnan(pred).any().item()}")
print(f"  pred 含 Inf? {torch.isinf(pred).any().item()}")
print(f"  pred 范围: [{pred.min():.4f}, {pred.max():.4f}]")

# ── 5. 损失计算 ───────────────────────────────────────────────────────────────
print("\n[步骤5] 损失计算...")
criterion = PredictionLoss(alpha=0.5)
loss = criterion(pred, y)
print(f"  loss = {loss.item()}")
print(f"  loss 是 NaN? {torch.isnan(loss).item()}")

if torch.isnan(loss):
    # 细化查找哪个环节出了 NaN
    print("\n  定位 NaN 来源:")
    print(f"  y 含 NaN? {torch.isnan(y).any().item()}")
    # 检查模型各层输出
    # 输入嵌入
    h = x.unsqueeze(-1)
    h = model.input_proj(h)
    print(f"  after input_proj NaN? {torch.isnan(h).any().item()}  "
          f"range=[{h.min():.3f}, {h.max():.3f}]")
    h = h + model.pos_emb[:, :window_size]
    print(f"  after pos_emb NaN?   {torch.isnan(h).any().item()}  "
          f"range=[{h.min():.3f}, {h.max():.3f}]")
    # encoder block 1
    h = model.encoder.blocks[0](h)
    print(f"  after block[0] NaN?  {torch.isnan(h).any().item()}  "
          f"range=[{h.min():.3f}, {h.max():.3f}]")

print("\n" + "=" * 60)
print("  诊断完成！把上面的输出贴给 Claude。")
print("=" * 60)
