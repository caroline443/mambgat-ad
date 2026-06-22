"""
SMAP / MSL 数据集下载脚本

直接从 GitHub LFS 地址下载 .npy 文件，无需 git-lfs，无需克隆 Telemanom。

用法：
  python download_data.py           # 下载 SMAP + MSL + 标签文件
  python download_data.py --smap    # 只下载 SMAP
  python download_data.py --msl     # 只下载 MSL

下载完成后目录结构：
  datasets/
    data/
      train/   P-1.npy  S-1.npy  ...
      test/    P-1.npy  S-1.npy  ...
    labeled_anomalies.csv
"""

import argparse
import os
import urllib.request
from pathlib import Path

# GitHub LFS 直链（绕过 git-lfs，直接 HTTP 下载）
BASE = "https://media.githubusercontent.com/media/khundman/telemanom/master"

SMAP_CHANNELS = [
    "P-1","S-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7","E-8","E-9",
    "E-10","E-11","E-12","E-13","A-1","D-1","P-2","P-3","D-2","D-3","D-4",
    "A-2","A-3","A-4","G-1","G-2","D-5","D-6","D-7","F-1","P-4","G-3",
    "T-1","T-2","D-8","D-9","F-2","G-4","T-3","D-11","D-12","B-1","G-6",
    "G-7","F-3","D-13","P-7","R-1","A-5","A-6","A-7","D-14","D-15","D-16",
]

MSL_CHANNELS = [
    "M-6","M-1","M-2","S-2","P-10","T-4","T-5","F-7","M-3","M-4","M-5",
    "P-15","C-1","C-2","T-12","T-13","F-4","F-5","D-16","M-7","F-6","T-9",
    "P-11","D-9","T-8","D-5","F-1",
]


def download_file(url: str, dest: Path, desc: str = ""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [跳过] {dest.name} 已存在")
        return
    try:
        print(f"  下载 {desc or dest.name} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"✓ ({size_kb} KB)")
    except Exception as e:
        print(f"✗ 失败: {e}")
        if dest.exists():
            dest.unlink()


def download_channels(channels: list, out_dir: Path, split: str):
    print(f"\n── {split}/ ({len(channels)} 个通道) ──")
    for ch in channels:
        url  = f"{BASE}/data/{split}/{ch}.npy"
        dest = out_dir / split / f"{ch}.npy"
        download_file(url, dest, ch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smap", action="store_true", help="只下载 SMAP")
    parser.add_argument("--msl",  action="store_true", help="只下载 MSL")
    parser.add_argument("--out",  default="datasets", help="输出目录（默认 datasets/）")
    args = parser.parse_args()

    # 默认两个都下
    do_smap = args.smap or (not args.smap and not args.msl)
    do_msl  = args.msl  or (not args.smap and not args.msl)

    out_dir = Path(args.out) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  MambGAT-AD 数据下载")
    print(f"  输出目录: {out_dir.resolve()}")
    print("=" * 50)

    if do_smap:
        print("\n▶ SMAP（NASA 土壤湿度卫星，55 通道）")
        download_channels(SMAP_CHANNELS, out_dir, "train")
        download_channels(SMAP_CHANNELS, out_dir, "test")

    if do_msl:
        print("\n▶ MSL（好奇号火星车，27 通道）")
        download_channels(MSL_CHANNELS, out_dir, "train")
        download_channels(MSL_CHANNELS, out_dir, "test")

    # 标签文件
    print("\n── 标签文件 ──")
    label_url  = f"{BASE}/labeled_anomalies.csv"
    label_dest = Path(args.out) / "labeled_anomalies.csv"
    download_file(label_url, label_dest, "labeled_anomalies.csv")

    # 统计
    npy_files = list(out_dir.rglob("*.npy"))
    print(f"\n✅ 完成！共下载 {len(npy_files)} 个 .npy 文件")
    print(f"   目录: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
