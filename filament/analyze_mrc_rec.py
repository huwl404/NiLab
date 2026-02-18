#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析同一数据集中 *_convmap.mrc (emClarity CC convolution map) 与 *.rec (断层图重建) 的信号关系，
便于决定纤维拟合时使用哪个体积或两者结合使用。

用法（在 filament 目录或项目根目录）：
  python filament/analyze_mrc_rec.py filament/data/cilia_106003_1_bin6
  python filament/analyze_mrc_rec.py filament/data  # 分析 data 下所有成对文件
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import mrcfile
except ImportError:
    print("请安装 mrcfile: pip install mrcfile", file=sys.stderr)
    sys.exit(1)


def find_pairs(data_dir: Path):
    """找出 (rec, convmap_mrc) 成对文件。"""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {data_dir}")
    recs = list(data_dir.glob("*.rec"))
    pairs = []
    for rec in recs:
        base = rec.stem
        conv = data_dir / f"{base}_convmap.mrc"
        if conv.exists():
            pairs.append((rec, conv))
        else:
            print(f"未找到对应 convmap: {rec.name} -> {base}_convmap.mrc", file=sys.stderr)
    return pairs


def load_volume(path: Path):
    """加载 MRC/REC 体积，返回 (data, voxel_size_angstrom)。mrcfile 中 data 为 (z,y,x)。"""
    with mrcfile.open(path, permissive=True, mode="r") as m:
        data = np.asarray(m.data, dtype=np.float32)
        h = m.header
        # 体素尺寸：通常从 header 的 cell 和 nx,ny,nz 得到
        try:
            cell = [h["cella"]["x"], h["cella"]["y"], h["cella"]["z"]]
            nxyz = [h["nx"], h["ny"], h["nz"]]
            voxel = [c / n if n else 0 for c, n in zip(cell, nxyz)]
            voxel_ang = voxel  # 若单位已是 Å；若为 nm 则乘 10
        except Exception:
            voxel_ang = [1.0, 1.0, 1.0]
    return data, voxel_ang


def analyze_pair(rec_path: Path, mrc_path: Path, sample_roi: bool = True):
    """比较一对 rec 与 convmap.mrc：形状、数值范围、相关性（整体或 ROI）。"""
    print(f"\n{'='*60}")
    print(f"REC:  {rec_path.name}")
    print(f"MRC:  {mrc_path.name}")
    print("=" * 60)

    rec_data, rec_vox = load_volume(rec_path)
    mrc_data, mrc_vox = load_volume(mrc_path)

    # 形状
    print(f"\n形状: REC {rec_data.shape} (Z,Y,X), MRC {mrc_data.shape} (Z,Y,X)")
    if rec_data.shape != mrc_data.shape:
        print("警告: 形状不一致，无法逐体素比较。")
        return

    # 数值范围
    print(f"\nREC  - min: {rec_data.min():.4f}, max: {rec_data.max():.4f}, mean: {rec_data.mean():.4f}, std: {rec_data.std():.4f}")
    print(f"MRC  - min: {mrc_data.min():.4f}, max: {mrc_data.max():.4f}, mean: {mrc_data.mean():.4f}, std: {mrc_data.std():.4f}")

    # 直方图简要（分位数）
    for name, arr in [("REC", rec_data), ("MRC", mrc_data)]:
        q = np.nanpercentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        print(f"{name} 分位数: 0%={q[0]:.4f}, 1%={q[1]:.4f}, 50%={q[4]:.4f}, 99%={q[7]:.4f}, 100%={q[8]:.4f}")

    # 线性相关（整体）
    rec_flat = rec_data.ravel()
    mrc_flat = mrc_data.ravel()
    # 下采样以节省内存
    step = max(1, len(rec_flat) // 1_000_000)
    r, m = rec_flat[::step], mrc_flat[::step]
    valid = np.isfinite(r) & np.isfinite(m)
    if valid.sum() > 10:
        corr = np.corrcoef(r[valid], m[valid])[0, 1]
        print(f"\n整体线性相关系数 (Pearson): {corr:.4f}")
    else:
        print("\n有效点过少，未计算相关系数。")

    # 中心 ROI 相关性（避免边界差异）
    if sample_roi and all(s > 20 for s in rec_data.shape):
        z, y, x = rec_data.shape
        z0, z1 = z // 4, 3 * z // 4
        y0, y1 = y // 4, 3 * y // 4
        x0, x1 = x // 4, 3 * x // 4
        rr = rec_data[z0:z1, y0:y1, x0:x1].ravel()[::step]
        mm = mrc_data[z0:z1, y0:y1, x0:x1].ravel()[::step]
        valid_roi = np.isfinite(rr) & np.isfinite(mm)
        if valid_roi.sum() > 10:
            corr_roi = np.corrcoef(rr[valid_roi], mm[valid_roi])[0, 1]
            print(f"中心 ROI 线性相关系数: {corr_roi:.4f}")

    # 建议
    print("\n建议:")
    print("  - convmap.mrc 为 cross-correlation map，通常更突出模板匹配位置，适合用于纤维中心线追踪。")
    print("  - .rec 为原始重建密度，信号连续但可能噪声更大；若纤维与背景对比度足够也可使用。")
    print("  - 若两者相关系数较高，可任选其一；若差异大(-0.0213)，建议纤维拟合优先使用 _convmap.mrc。")
    print("  - 纤维不连续时可考虑对体积做侵蚀或骨架化后再追踪。")


def main():
    ap = argparse.ArgumentParser(description="分析 convmap.mrc 与 .rec 信号关系")
    ap.add_argument("path", nargs="?", default="data", help="单个前缀路径（如 data/cilia_106003_1_bin6）或 data 目录")
    ap.add_argument("--no-roi", action="store_true", help="不计算中心 ROI 相关")
    args = ap.parse_args()

    path = Path(args.path)
    if path.suffix or not path.is_dir():
        # 视为单个前缀
        data_dir = path.parent
        base = path.name
        rec = data_dir / f"{base}.rec"
        mrc = data_dir / f"{base}_convmap.mrc"
        if not rec.exists():
            rec = path.with_suffix(".rec")
            mrc = path.parent / f"{path.stem}_convmap.mrc"
        if rec.exists() and mrc.exists():
            analyze_pair(rec, mrc, sample_roi=not args.no_roi)
        else:
            print(f"未找到成对文件: {rec} / {mrc}", file=sys.stderr)
            sys.exit(1)
    else:
        pairs = find_pairs(path)
        if not pairs:
            print("未找到任何 rec/convmap 对。", file=sys.stderr)
            sys.exit(1)
        for rec, mrc in pairs:
            analyze_pair(rec, mrc, sample_roi=not args.no_roi)


if __name__ == "__main__":
    main()
