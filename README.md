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

### 1. 安装环境（Windows + Anaconda）

```batch
conda create -n mambgat python=3.10 -y
conda activate mambgat

# 先安装 PyTorch（根据你的 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 准备数据（SMAP / MSL）

```batch
:: 下载 Telemanom 数据集
git clone https://github.com/khundman/telemanom
:: 将数据复制到本项目
mkdir datasets
xcopy /E telemanom\data datasets\data\
copy telemanom\labeled_anomalies.csv datasets\

:: 目录结构应为：
:: datasets/
::   data/
::     train/  P-1.npy  S-1.npy  ...
::     test/   P-1.npy  S-1.npy  ...
::   labeled_anomalies.csv
```

### 3. 验证代码（无需真实数据）

```batch
python sanity_check.py
```

### 4. 开始训练

```batch
:: 训练 SMAP
python train.py --config config/smap.yaml

:: 训练 MSL
python train.py --config config/smap.yaml --dataset msl

:: 自定义参数
python train.py --config config/smap.yaml --epochs 50 --lr 5e-4
```

### 5. 评估

```batch
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

## 显存估算（A4000 16GB）

| 配置 | 参数量 | 显存占用（估算） |
|------|--------|----------------|
| d_model=64, n_blocks=2 | ~1.5M | ~2-3 GB |
| d_model=128, n_blocks=3 | ~5M | ~5-7 GB |
| d_model=256, n_blocks=4 | ~18M | ~10-12 GB |

A4000 16GB 跑 d_model=128 完全没问题，留有充足余量。

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
