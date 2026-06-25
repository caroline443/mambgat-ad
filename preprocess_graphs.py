"""
preprocess_graphs.py — 离线预计算 DTW 动态图（参考 ContrastAD graph_info.py）

对训练集和测试集的每个滑动窗口，计算三个图：
  g1:      最不相似的图快照（KL 散度最大的图对中的 g1）
  g2:      最不相似的图对中的 g2
  avg_adj: 其余快照的均值（锚图）

输出目录结构：
  datasets/AT/<DATASET>/graphs/
      train_g1.npy    (n_train_windows, N, N)  float32
      train_g2.npy
      train_avg.npy
      test_g1.npy     (n_test_windows,  N, N)  float32
      test_g2.npy
      test_avg.npy

用法：
  pip install dtaidistance joblib          # 服务器上运行
  python preprocess_graphs.py --dataset smap --data_dir ./datasets/AT/SMAP
  python preprocess_graphs.py --dataset msl  --data_dir ./datasets/AT/MSL
  python preprocess_graphs.py --dataset smd  --data_dir ./datasets/AT/SMD

参数：
  --train_step  训练窗口步长，与 train.py 保持一致（SMAP=5, MSL=1, SMD=10）
  --n_jobs      并行进程数（建议 = CPU 核数，服务器上可用 8-16）
  --num_snap    每个窗口切分的快照数（默认 10，与 ContrastAD 一致）
"""

import argparse
import math
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.special import rel_entr
from tqdm import tqdm

# ── DTW 后端：优先用 dtaidistance（快 10-100x），不可用时退化为相关性距离 ──
try:
    from dtaidistance import dtw as _dtw
    def _pairwise_dist(segment: np.ndarray) -> np.ndarray:
        """segment: (T, N) → 返回 (N, N) DTW 距离矩阵"""
        return _dtw.distance_matrix_fast(segment.T.astype(np.float64))
    print("[Graph] dtaidistance 可用 → 使用精确 DTW 距离")
except ImportError:
    def _pairwise_dist(segment: np.ndarray) -> np.ndarray:
        """
        fallback：用相关性距离代替 DTW。
        dist(i,j) = 1 - |corr(i,j)|，同样能刻画两通道间时序相似度。
        """
        T, N = segment.shape
        if T < 2:
            return np.zeros((N, N), dtype=np.float64)
        # 标准化，防止常数通道报错
        s = segment - segment.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(s, axis=0, keepdims=True) + 1e-8
        s = s / norm
        corr = s.T @ s / max(T - 1, 1)           # (N, N) 相关系数矩阵
        corr = np.clip(corr, -1, 1)
        return 1.0 - np.abs(corr)                  # 距离：越接近 0 越相似
    print("[Graph] dtaidistance 未安装 → 使用相关性距离（fallback）")
    print("        建议安装: pip install dtaidistance")


# ─────────────────────────────────────────────────────────────────────────────
# 图构建工具函数（直接参考 ContrastAD graph_info.py）
# ─────────────────────────────────────────────────────────────────────────────

def _expected_edges(n_nodes: int) -> int:
    """ContrastAD 的调和级数边数启发式"""
    h_sum = e_sum = 0.0
    for i in range(1, n_nodes + 1):
        h_sum += 1.0 / i
        e_sum += 1.0 / h_sum
    return 2 * math.ceil(e_sum) + n_nodes


def _gen_binary_adj(segment: np.ndarray, expected_edges: int,
                    from_top: bool = True) -> np.ndarray:
    """
    对一个时间段计算 DTW/相关性距离矩阵，选 top/bottom 边生成二值邻接矩阵。

    Args:
        segment:        (T_seg, N) 时序片段
        expected_edges: 保留的边数
        from_top:       True=保留最大距离边（最不相似），False=保留最小距离边
    Returns:
        (N, N) float32 二值邻接矩阵
    """
    N = segment.shape[1]
    ds = _pairwise_dist(segment)                    # (N, N)
    result = np.zeros((N, N), dtype=np.float32)

    if from_top:
        flat = ds.flatten()
        top_idx = np.argsort(flat)[-expected_edges:]
        rows, cols = np.unravel_index(top_idx, ds.shape)
        result[rows, cols] = 1.0
    else:
        # 排除对角线 0
        flat = ds.flatten()
        nz_mask = flat > 0
        nz_vals = flat[nz_mask]
        nz_idx  = np.where(nz_mask)[0]
        edge_cnt = min(expected_edges, len(nz_vals))
        small_idx = nz_idx[np.argsort(nz_vals)[:edge_cnt]]
        rows, cols = np.unravel_index(small_idx, ds.shape)
        result[rows, cols] = 1.0

    return result


def _sym_kl(p: np.ndarray, q: np.ndarray) -> float:
    """对称 KL 散度"""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return float(np.sum(rel_entr(p, q)) + np.sum(rel_entr(q, p)))


def _degree_dist(adj: np.ndarray) -> np.ndarray:
    """从邻接矩阵计算归一化度分布"""
    deg = adj.sum(axis=1).astype(np.float32)
    total = deg.sum()
    return deg / total if total > 0 else np.full_like(deg, 1.0 / len(deg))


def compute_window_graphs(window: np.ndarray, num_snap: int = 10
                           ) -> tuple:
    """
    对单个窗口计算 g1, g2, avg_adj。

    Args:
        window:   (W, N) 归一化后的时序窗口
        num_snap: 快照数（默认 10，与 ContrastAD 一致）
    Returns:
        (g1, g2, avg_adj)，各 (N, N) float32
    """
    T, N = window.shape
    seg_len = T // num_snap
    if seg_len < 1:
        zeros = np.zeros((N, N), dtype=np.float32)
        return zeros, zeros, zeros

    expected = _expected_edges(N)
    adjs = []
    for s in range(num_snap):
        start = s * seg_len
        end   = start + seg_len if s < num_snap - 1 else T
        seg   = window[start:end]
        adjs.append(_gen_binary_adj(seg, expected, from_top=True))

    # 找 KL 散度最大的图对
    degs = [_degree_dist(a) for a in adjs]
    best_kl, best_pair = -1.0, (0, 1)
    for i, j in combinations(range(len(adjs)), 2):
        kl = _sym_kl(degs[i], degs[j])
        if kl > best_kl:
            best_kl, best_pair = kl, (i, j)

    i, j = best_pair
    g1, g2 = adjs[i], adjs[j]
    remaining = [a for k, a in enumerate(adjs) if k not in best_pair]
    avg_adj = np.mean(remaining, axis=0).astype(np.float32) if remaining \
              else np.zeros((N, N), dtype=np.float32)

    return g1, g2, avg_adj


# ─────────────────────────────────────────────────────────────────────────────
# 批量预计算
# ─────────────────────────────────────────────────────────────────────────────

def precompute_graphs(
    data: np.ndarray,        # (T, N) 归一化后的时序
    window_size: int,
    step: int,
    num_snap: int,
    n_jobs: int,
    desc: str = "",
) -> tuple:
    """
    对 data 中所有滑窗计算图，返回 (g1_arr, g2_arr, avg_arr)，
    各形状 (n_windows, N, N)。
    """
    T, N = data.shape
    indices = list(range(0, T - window_size, step))
    n_windows = len(indices)
    print(f"  {desc}: {n_windows:,} 个窗口 × {num_snap} 快照 × {N} 通道")

    def _one(idx):
        w = data[idx : idx + window_size]
        return compute_window_graphs(w, num_snap)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one)(i) for i in tqdm(indices, desc=f"  [{desc}]", ncols=72)
    )

    g1_arr  = np.stack([r[0] for r in results], axis=0)
    g2_arr  = np.stack([r[1] for r in results], axis=0)
    avg_arr = np.stack([r[2] for r in results], axis=0)
    return g1_arr, g2_arr, avg_arr


# ─────────────────────────────────────────────────────────────────────────────
# 数据归一化（与 train.py 保持一致）
# ─────────────────────────────────────────────────────────────────────────────

def normalize(train: np.ndarray, test: np.ndarray) -> tuple:
    mean = train.mean(axis=0, keepdims=True)
    std  = np.where(train.std(axis=0, keepdims=True) < 1e-4,
                    1.0, train.std(axis=0, keepdims=True))
    tr = np.clip((train - mean) / std, -10, 10).astype(np.float32)
    te = np.clip((test  - mean) / std, -10, 10).astype(np.float32)
    return tr, te


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="smap",
                        choices=["smap", "msl", "smd"])
    parser.add_argument("--data_dir",   default="./datasets/AT/SMAP")
    parser.add_argument("--window_size",type=int, default=100)
    parser.add_argument("--train_step", type=int, default=5,
                        help="训练集滑窗步长（SMAP=5, MSL=1, SMD=10）")
    parser.add_argument("--test_step",  type=int, default=1)
    parser.add_argument("--num_snap",   type=int, default=10,
                        help="每窗口切分的快照数（与 ContrastAD 一致）")
    parser.add_argument("--n_jobs",     type=int, default=4,
                        help="并行线程数")
    args = parser.parse_args()

    ds   = args.dataset.upper()
    ddir = Path(args.data_dir)

    print(f"\n{'='*55}")
    print(f"  预计算 DTW 图  |  数据集={ds}")
    print(f"  train_step={args.train_step}  test_step={args.test_step}")
    print(f"  num_snap={args.num_snap}  n_jobs={args.n_jobs}")
    print(f"{'='*55}\n")

    # ── 加载数据 ──────────────────────────────────────────────────
    train_raw = np.load(ddir / f"{ds}_train.npy").astype(np.float32)
    test_raw  = np.load(ddir / f"{ds}_test.npy" ).astype(np.float32)
    print(f"[Data] 训练={train_raw.shape}  测试={test_raw.shape}")

    train_np, test_np = normalize(train_raw, test_raw)

    # ── 输出目录 ──────────────────────────────────────────────────
    out_dir = ddir / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] → {out_dir}\n")

    # ── 计算训练集图 ──────────────────────────────────────────────
    t0 = time.time()
    print("[1/2] 计算训练集图...")
    tr_g1, tr_g2, tr_avg = precompute_graphs(
        train_np, args.window_size, args.train_step,
        args.num_snap, args.n_jobs, desc="train"
    )
    np.save(out_dir / "train_g1.npy",  tr_g1)
    np.save(out_dir / "train_g2.npy",  tr_g2)
    np.save(out_dir / "train_avg.npy", tr_avg)
    t1 = time.time()
    print(f"  ✓ 训练集图已保存  shape={tr_g1.shape}  耗时={t1-t0:.0f}s\n")

    # ── 计算测试集图 ──────────────────────────────────────────────
    print("[2/2] 计算测试集图...")
    te_g1, te_g2, te_avg = precompute_graphs(
        test_np, args.window_size, args.test_step,
        args.num_snap, args.n_jobs, desc="test"
    )
    np.save(out_dir / "test_g1.npy",  te_g1)
    np.save(out_dir / "test_g2.npy",  te_g2)
    np.save(out_dir / "test_avg.npy", te_avg)
    t2 = time.time()
    print(f"  ✓ 测试集图已保存  shape={te_g1.shape}  耗时={t2-t1:.0f}s")

    total = t2 - t0
    print(f"\n{'='*55}")
    print(f"  完成！总耗时 {total/60:.1f} 分钟")
    print(f"  文件位置: {out_dir}")
    print(f"  各文件形状: train {tr_g1.shape}, test {te_g1.shape}")
    print(f"  磁盘占用约: {(tr_g1.nbytes*3 + te_g1.nbytes*3)/1e9:.2f} GB")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
