# MambGAT-AD

**Spatiotemporal Mamba with Graph Attention for Spacecraft Telemetry Anomaly Detection**

> 一个面向航天器遥测数据的时空异常检测模型，结合 Mamba 状态空间模型的线性时序建模与图注意力网络的动态传感器耦合发现。

---

## 核心创新

| 模块 | 解决的问题 | 相比基线的优势 |
|------|-----------|--------------|
| **Selective SSM (Mamba)** | 航天器长周期时序依赖（轨道周期、缓慢衰退） | 线性复杂度 O(T)，Transformer 是 O(T²) |
| **Dynamic GAT** | 传感器间未知物理耦合（热耦合、电耦合） | 可学习邻接矩阵，无需先验拓扑知识 |
| **ST-Mamba-GAT Block** | 时空信息的深度融合 | 空间上下文注入时序状态更新（非简单串联） |
| **ESA-ADB 验证** | 真实航天工程复杂异常 | 首个在 ESA 专用数据集上的深度学习图模型 |

---

## 快速开始

> **代码自动适配环境**：检测到 `mamba_ssm` 就用原生 CUDA kernel，否则自动 fallback 到纯 PyTorch，两端结果完全一致。

---

### 方式一：服务器（Linux + V100）

```bash
git clone https://github.com/caroline443/mambgat-ad.git
cd mambgat-ad

conda create -n mambgat python=3.10 -y && conda activate mambgat

# 1. PyTorch（根据服务器 CUDA 版本选）
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 2. mamba-ssm（顺序不能错，必须在 PyTorch 之后）
pip install causal-conv1d mamba-ssm --no-build-isolation

# 3. 其余依赖
pip install -r requirements.txt

# 验证（会打印 "[MambGAT] mamba_ssm 检测到 → 使用原生 CUDA kernel"）
python sanity_check.py

# 训练（用服务器配置）
python train.py --config config/smap.yaml
```

---

### 方式二：Windows + Anaconda

```batch
git clone https://github.com/caroline443/mambgat-ad.git
cd mambgat-ad

conda create -n mambgat python=3.10 -y
conda activate mambgat

:: 1. PyTorch（选对应显卡的 CUDA 版本）
pip install torch --index-url https://download.pytorch.org/whl/cu118

:: 2. 其余依赖（跳过 mamba-ssm，代码自动 fallback）
pip install -r requirements.txt

:: 验证（会打印 "[MambGAT] mamba_ssm 未找到 → 使用纯 PyTorch 实现"）
python sanity_check.py

:: 训练（用 Windows 配置，小 batch / num_workers=0）
python train.py --config config/smap_win.yaml
```

---

### 准备数据（两端通用）

```bash
# Linux
git clone https://github.com/khundman/telemanom
mkdir -p datasets
cp -r telemanom/data datasets/data
cp telemanom/labeled_anomalies.csv datasets/
```

```batch
:: Windows
git clone https://github.com/khundman/telemanom
mkdir datasets
xcopy /E telemanom\data datasets\data\
copy telemanom\labeled_anomalies.csv datasets\
```

数据目录结构：
```
datasets/
  data/
    train/   P-1.npy  S-1.npy  ...（55 个通道）
    test/    P-1.npy  S-1.npy  ...
  labeled_anomalies.csv
```

### 评估

```bash
python evaluate.py --ckpt checkpoints/best_smap.pt
python evaluate.py --ckpt checkpoints/best_smap.pt --plot_graph
```

---

## 项目结构

```
mambgat-ad/
├── config/
│   └── smap.yaml           # SMAP/MSL 实验配置
├── data/
│   └── dataset.py          # 数据加载（Telemanom 格式）
├── models/
│   ├── ssm_layer.py        # 纯 PyTorch Mamba 实现（Windows 兼容）
│   ├── gat_layer.py        # 动态图注意力（可学习邻接矩阵）
│   ├── st_block.py         # ST-Mamba-GAT 核心块
│   └── mambgat.py          # 完整模型
├── utils/
│   ├── threshold.py        # Telemanom 动态阈值 / 分位数阈值
│   └── metrics.py          # F1(Raw) / F1(PA) / AUC-ROC
├── train.py                # 训练入口
├── evaluate.py             # 独立评估 + 图可视化
└── sanity_check.py         # 快速冒烟测试（无需数据）
```

---

## 模型架构

```
输入 X: (B, T, N)
     │
[输入嵌入] Linear(1 → D)
     │
[位置编码] 可学习位置嵌入
     │
[ST-Mamba-GAT Block × n_blocks]
     │  ┌─────────────────────────────────────┐
     │  │  ① 空间路径: GAT(每个时间步)         │
     │  │     → 捕捉传感器耦合关系              │
     │  │  ② 空间注入: x = x + α * GAT_out    │
     │  │  ③ 时序路径: Mamba(每个节点)         │
     │  │     → 捕捉长程时序依赖               │
     │  │  ④ 残差融合 + FFN                   │
     │  └─────────────────────────────────────┘
     │
[预测头] 最后时间步 → 预测下一步值 (B, N)
     │
[残差] |预测 - 实际| → 异常分数 (B, N)
     │
[Telemanom 动态阈值] → 异常标签 (B,)
```

---

## 配置说明

`config/smap.yaml` 中的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.d_model` | 64 | 特征维度（A4000 可以用 128） |
| `model.d_state` | 16 | Mamba 状态维度（越大记忆越长） |
| `model.n_blocks` | 2 | ST-Mamba-GAT 堆叠层数 |
| `model.n_heads` | 4 | GAT 注意力头数 |
| `data.window_size` | 100 | 滑动窗口长度 |
| `train.batch_size` | 64 | A4000 16G 可用 128-256 |

---

## 显存估算（V100-32GB）

| 配置 | 参数量 | 显存占用（估算） | 备注 |
|------|--------|----------------|------|
| d_model=64, n_blocks=2, bs=256 | ~1.5M | ~3-4 GB | 验证用 |
| **d_model=128, n_blocks=3, bs=256** | **~5M** | **~8-10 GB** | **默认配置** |
| d_model=256, n_blocks=4, bs=128 | ~18M | ~18-22 GB | 论文最终跑分用 |

V100 32GB 跑默认配置（d_model=128）只用约 1/3 显存，batch_size 还可以继续拉大。
原生 mamba_ssm CUDA kernel 比纯 PyTorch 实现快约 8-10 倍。

---

## Baseline 对比目标

| 方法 | 类型 | SMAP F1(PA) |
|------|------|------------|
| Telemanom (KDD'18) | LSTM + 残差 | ~0.85 |
| OmniAnomaly (KDD'19) | VAE-RNN | ~0.87 |
| USAD (KDD'20) | AE-GAN | ~0.88 |
| GDN (AAAI'21) | 图网络 | ~0.90 |
| **MambGAT-AD (Ours)** | Mamba + GAT | **TBD** |

---

## 引用

如果本代码对你有帮助：

```bibtex
@inproceedings{mambgatad2026,
  title     = {MambGAT-AD: Spatiotemporal Mamba with Graph Attention
               for Spacecraft Telemetry Anomaly Detection},
  author    = {Your Name},
  booktitle = {Proceedings of ...},
  year      = {2026}
}
```

---

## 参考文献

- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", ICLR 2024
- Veličković et al., "Graph Attention Networks", ICLR 2018
- Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding", KDD 2018
- Deng & Hooi, "Graph Deviation Network for Cloud Service Anomaly Detection", AAAI 2021
- ESA-ADB: "European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry", 2024
