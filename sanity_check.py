"""
快速冒烟测试 — 在下载真实数据前验证模型代码正确性

运行环境：Linux 服务器，已安装 mamba-ssm

用法：
  python sanity_check.py

成功输出示例：
  [OK] mamba_ssm 导入成功
  [OK] 前向传播通过  pred=(4, 55, 1)  score=(4, 55)
  [OK] 反向传播通过  loss=0.1234
  [OK] 参数量=2,345,678
  [OK] 所有测试通过！模型代码正常，可以准备真实数据了。
"""

import torch
import numpy as np

def main():
    print("=" * 50)
    print("  MambGAT-AD 快速冒烟测试")
    print("=" * 50)

    # ── 检查 mamba_ssm ────────────────────────────────────────────
    try:
        import mamba_ssm
        print(f"[OK] mamba_ssm 导入成功  版本={mamba_ssm.__version__}")
    except ImportError as e:
        print(f"[FAIL] mamba_ssm 未安装: {e}")
        print("       请运行：pip install causal-conv1d mamba-ssm --no-build-isolation")
        return

    from models import MambGATAD, PredictionLoss

    # 模拟 SMAP 的配置：55 个通道，窗口长度 100
    N_CHANNELS  = 55
    WINDOW_SIZE = 100
    BATCH_SIZE  = 4
    D_MODEL     = 32   # 测试时用小模型加速

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    # ── 构建模型 ──────────────────────────────────────────────────
    model = MambGATAD(
        n_channels  = N_CHANNELS,
        window_size = WINDOW_SIZE,
        d_model     = D_MODEL,
        n_blocks    = 2,
        n_heads     = 4,
        d_state     = 8,
        d_conv      = 4,
        expand      = 2,
        pred_len    = 1,
        dropout     = 0.0,
    ).to(device)

    # ── 前向传播 ──────────────────────────────────────────────────
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, N_CHANNELS).to(device)
    pred, score = model(x)

    assert pred.shape  == (BATCH_SIZE, N_CHANNELS, 1), \
        f"pred 形状错误: {pred.shape}"
    assert score.shape == (BATCH_SIZE, N_CHANNELS), \
        f"score 形状错误: {score.shape}"
    print(f"[OK] 前向传播通过  pred={tuple(pred.shape)}  score={tuple(score.shape)}")

    # ── 反向传播 ──────────────────────────────────────────────────
    y = torch.randn(BATCH_SIZE, N_CHANNELS).to(device)
    criterion = PredictionLoss()
    loss = criterion(pred, y)
    loss.backward()
    print(f"[OK] 反向传播通过  loss={loss.item():.4f}")

    # ── 参数量 ────────────────────────────────────────────────────
    n_params = model.count_parameters()
    print(f"[OK] 参数量={n_params:,}")

    # ── 图结构提取 ────────────────────────────────────────────────
    adj = model.get_graph()
    assert adj.shape == (N_CHANNELS, N_CHANNELS), \
        f"邻接矩阵形状错误: {adj.shape}"
    print(f"[OK] 传感器耦合图提取成功  adj={tuple(adj.shape)}")

    # ── 阈值模块 ──────────────────────────────────────────────────
    from utils.threshold import PerChannelThreshold
    fake_train_errors = np.random.rand(500, N_CHANNELS)
    fake_test_errors  = np.random.rand(200, N_CHANNELS)
    thr = PerChannelThreshold(method="percentile", percentile=95.0)
    thr.fit(fake_train_errors)
    per_ch, global_pred = thr.predict(fake_test_errors)
    assert per_ch.shape     == (200, N_CHANNELS)
    assert global_pred.shape == (200,)
    print(f"[OK] 阈值模块正常  异常率={global_pred.mean():.2%}")

    # ── 指标模块 ──────────────────────────────────────────────────
    from utils.metrics import evaluate_anomaly
    y_true = (np.random.rand(200) > 0.85).astype(int)
    metrics = evaluate_anomaly(y_true, global_pred, y_score=fake_test_errors.mean(1))
    print(f"[OK] 指标模块正常  f1_pa={metrics.get('f1_pa', 'N/A'):.4f}")

    print("\n[OK] 所有测试通过！模型代码正常，可以准备真实数据了。")
    print("=" * 50)
    print("\n下一步：")
    print("  1. git clone https://github.com/khundman/telemanom")
    print("  2. 将 telemanom/data/ 复制到本项目 datasets/ 目录")
    print("  3. python train.py --config config/smap.yaml")


if __name__ == "__main__":
    main()
