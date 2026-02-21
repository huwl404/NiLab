#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于二维切片互相关的纤维追踪：
fiber start/end .star + .rec tomogram -> fiber tracing -> export .axm + particles .star

设计要点：
1) 不裁剪 ROI，直接在原始体积上进行切片采样与互相关追踪。
2) 纤维截面按正圆近似，采样窗口由 radius + margin 控制。
3) 提供体积增强（对比度与连续性）以应对断续纤维。
4) --debug 打开时，将中间 mrc 输出到输入 rec 同目录。
5) 支持按纤维多进程并行追踪。
"""
from __future__ import annotations

import argparse
import atexit
import multiprocessing
import os
import sys
from multiprocessing import shared_memory
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splprep, splev

try:
    from skimage.registration import phase_cross_correlation
except ImportError:
    phase_cross_correlation = None

try:
    import mrcfile
except ImportError:
    mrcfile = None

try:
    import starfile
except ImportError:
    starfile = None


def read_star_fibers(star_path: Path) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], bool]:
    if starfile is None:
        raise ImportError("please install starfile: pip install starfile")
    data = starfile.read(star_path, always_dict=True)
    df = next(iter(data.values())) if isinstance(data, dict) else data

    def col(name: str) -> Optional[str]:
        for k in (f"_rln{name}", f"rln{name}", name):
            if k in df.columns:
                return k
        return None

    xc, yc, zc = col("CoordinateX"), col("CoordinateY"), col("CoordinateZ")
    ra, ta, pa = col("AngleRot"), col("AngleTilt"), col("AnglePsi")
    if not all((xc, yc, zc)):
        raise ValueError("input .star missing CoordinateX/Y/Z columns")

    coords = df[[xc, yc, zc]].to_numpy(dtype=np.float32)
    angles = np.zeros((len(df), 3), dtype=np.float32)
    if all((ra, ta, pa)):
        angles = df[[ra, ta, pa]].to_numpy(dtype=np.float32)

    fibers: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(0, len(coords) - 1, 2):
        fibers.append((coords[i], coords[i + 1], angles[i], angles[i + 1]))
    return fibers, (len(coords) % 2 == 1)


def parse_fiber_index_list(spec: Optional[str], n_fibers: int) -> List[int]:
    if spec is None or str(spec).strip() == "":
        return list(range(n_fibers))
    out = set()
    for part in str(spec).split(","):
        s = part.strip()
        if not s:
            continue
        if "-" in s and s.count("-") == 1:
            a, b = s.split("-", 1)
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            for i in range(min(lo, hi), max(lo, hi) + 1):
                if 0 <= i < n_fibers:
                    out.add(i)
        else:
            try:
                i = int(s)
            except ValueError:
                continue
            if 0 <= i < n_fibers:
                out.add(i)
    return sorted(out)


def load_volume(rec_path: Path) -> np.ndarray:
    if mrcfile is None:
        raise ImportError("please install mrcfile: pip install mrcfile")
    with mrcfile.open(rec_path, permissive=True, mode="r") as m:
        vol = np.asarray(m.data, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape={vol.shape}")
    return vol


def save_mrc(path: Path, data: np.ndarray, voxel_size: float = 1.0) -> None:
    if mrcfile is None:
        raise ImportError("please install mrcfile: pip install mrcfile")
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(np.asarray(data, dtype=np.float32))
        try:
            m.voxel_size = float(voxel_size)
        except Exception:
            pass


def normalize_volume(vol: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(vol, [low, high])
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    x = (vol - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def enhance_volume(
    vol: np.ndarray,
    voxel_size: float,
    gaussian_sigma: float,
    closing_size: int,
    debug_mrc_prefix: Optional[str],
) -> np.ndarray:
    # 互相关前建议做轻量增强；低对比/断续纤维直接做互相关容易退化成直线。
    x = normalize_volume(vol)
    if debug_mrc_prefix is not None:
        save_mrc(f"{debug_mrc_prefix}_1_normalize.mrc", x, voxel_size=voxel_size)

    if gaussian_sigma > 0:
        x = ndi.gaussian_filter(x, sigma=gaussian_sigma).astype(np.float32)
        if debug_mrc_prefix is not None:
            save_mrc(f"{debug_mrc_prefix}_2_gaussian.mrc", x, voxel_size=voxel_size)

    if closing_size > 1:
        s = int(closing_size)
        x = ndi.grey_closing(x, size=(s, s, s)).astype(np.float32)
        if debug_mrc_prefix is not None:
            save_mrc(f"{debug_mrc_prefix}_3_closing.mrc", x, voxel_size=voxel_size)

    return x


def _normalize(v: np.ndarray) -> np.ndarray:
    """normalize the vector"""
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _xyz_inside(shape_zyx: Tuple[int, int, int], p_xyz: np.ndarray) -> bool:
    """check if the point is inside the volume"""
    x, y, z = p_xyz
    zmax, ymax, xmax = shape_zyx
    return (0 <= x < xmax) and (0 <= y < ymax) and (0 <= z < zmax)


def _pick_axis_plane(direction_xyz: np.ndarray) -> Tuple[str, int, int]:
    """
    按起止点连线方向，固定本条纤维使用的“主轴 + 切面”。
    按前进方向选择轴对齐切面：
    - 主要沿 Z -> 用 XY（法向 z）
    - 主要沿 Y -> 用 XZ（法向 y）
    - 主要沿 X -> 用 YZ（法向 x）
    返回:
    - plane:  "xy" / "xz" / "yz"
    - axis:   主轴在 xyz 中的索引 (0/1/2)
    - sign:   主轴前进方向 (+1 或 -1)
    """
    d = np.abs(_normalize(direction_xyz))
    axis = int(np.argmax(d))
    sign = 1 if float(direction_xyz[axis]) >= 0 else -1
    if axis == 2:  # z-axis
        return "xy", 2, sign
    if axis == 1:  # y-axis
        return "xz", 1, sign
    return "yz", 0, sign  # x-axis


def sample_axis_slice(vol_zyx: np.ndarray, center_xyz: np.ndarray, plane: str, half: float, out_size: int) -> np.ndarray:
    """
    在体积中以 center 为中心，采样一张轴对齐 2D 切片。
    输出约定（用于 cc 的 row/col）：
    - xy: row=y, col=x
    - xz: row=z, col=x
    - yz: row=z, col=y
    """
    t = np.linspace(-half, half, out_size, dtype=np.float32)  # 生成一维坐标轴：-half, half, out_size 个点
    xx, yy = np.meshgrid(t, t, indexing="xy")  # 生成二维网格： out_size x out_size 的矩阵， xx 为列偏移网格，yy 为行偏移网格
    cx, cy, cz = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])

    if plane == "xy":  # 根据 plane 选择采样平面，告诉采样坐标
        x = cx + xx
        y = cy + yy
        z = np.full_like(x, cz, dtype=np.float32)
    elif plane == "xz":
        x = cx + xx
        z = cz + yy
        y = np.full_like(x, cy, dtype=np.float32)
    else:  # yz
        y = cy + xx
        z = cz + yy
        x = np.full_like(y, cx, dtype=np.float32)

    coords_zyx = np.stack([z, y, x], axis=0)
    sl = ndi.map_coordinates(vol_zyx, coords_zyx, order=1, mode="nearest")  # 使用一阶线性插值法对体积进行采样
    return sl.astype(np.float32)


def sample_axis_slice_merged(
    vol_zyx: np.ndarray,
    center_xyz: np.ndarray,
    plane: str,
    normal_axis: int,
    half: float,
    out_size: int,
    merge_step: int,
) -> np.ndarray:
    """
    沿切面法向前后各取 merge_step 张切片，与中心切片一起做均值。
    例如 merge_step=3 时，实际平均 7 张切片（-3..+3），
    可显著降低单层噪声，提高 phase cross-correlation 的稳定性。
    """
    acc = np.zeros((out_size, out_size), dtype=np.float32)
    for k in range(-merge_step, merge_step + 1):
        p = np.asarray(center_xyz, dtype=np.float32).copy()  # center point
        p[normal_axis] += float(k)  # move along the normal axis
        acc += sample_axis_slice(vol_zyx, p, plane, half, out_size)
    return acc / float(2 * merge_step + 1)


def trace_fiber_crosscorr(
    vol_zyx: np.ndarray,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    radius_px: float,
    merge_step: int,
    curve_points: int,
    curvature: float,
    margin_ratio: float,
    guide_weight: float,
    cc_upsample: int,
    debug_mrc_prefix: Optional[str],
    voxel_size: float,
) -> np.ndarray:
    """
    严格轴向追踪主函数（不做斜切面）：

    1) 根据起止点方向固定主轴（x/y/z 之一）和切面（yz/xz/xy 之一）
    2) 仅沿主轴每步前进 merge_step + 1 像素
    3) 在固定切面上做“多层平均后”的 2D 互相关
    4) 将 2D 平移映射到 3D 横向校正，轴向保持严格步进
    5) 最后用样条平滑并强制首尾点
    """
    if phase_cross_correlation is None:
        raise ImportError("phase cross correlation needs scikit-image: pip install scikit-image")

    start_xyz = np.asarray(start_xyz, dtype=np.float32)
    end_xyz = np.asarray(end_xyz, dtype=np.float32)
    direction = end_xyz - start_xyz
    total_len = float(np.linalg.norm(direction))
    if total_len < 1e-4:
        print("error: start and end points are too close")
        return np.vstack([start_xyz, end_xyz]).astype(np.float32)

    merge_step = max(0, int(merge_step))
    # 严格轴向步进：先固定主轴，再只沿该轴每次前进 merge_step + 1 像素。
    plane, normal_axis, axis_sign = _pick_axis_plane(direction)
    axis_step = axis_sign * (merge_step + 1)
    axis_span = abs(float(direction[normal_axis]))
    n_steps = max(2, int(axis_span / axis_step))

    # 单一 half：正圆窗口半径 + 额外 margin。
    half = max(1.0, radius_px) + max(1.0, radius_px * margin_ratio)
    out_size = int(np.ceil(2 * half)) + 1
    centers = [start_xyz.copy()]
    center = start_xyz.copy()

    for i in range(1, n_steps):
        t = i / float(n_steps)
        guide = start_xyz + t * (end_xyz - start_xyz) 

        pred = center.copy()
        pred[normal_axis] += axis_step  # e.g.: z-axis: pred[2] += axis_step
        if not _xyz_inside(vol_zyx.shape, pred):
            pred = np.clip(pred, [0, 0, 0], [vol_zyx.shape[2] - 1, vol_zyx.shape[1] - 1, vol_zyx.shape[0] - 1])

        # 前后点都使用“法向多层平均切片”做 cc。
        prev_slice = sample_axis_slice_merged(vol_zyx, center, plane, normal_axis, half, out_size, merge_step)
        cur_slice = sample_axis_slice_merged(vol_zyx, pred, plane, normal_axis, half, out_size, merge_step)
        if debug_mrc_prefix is not None:
            print(pred)
            # save_mrc(f"{debug_mrc_prefix}_{i}_slice.mrc", prev_slice, voxel_size=voxel_size)

        # 2D 子像素位移（row, col）。prev 当 reference、cur 当 moving。
        shift, error, _ = phase_cross_correlation(prev_slice, cur_slice, upsample_factor=max(1, int(cc_upsample)))
        # 将 2D shift 映射回 xyz（只作用于切面内两个横向轴）。
        corr_xyz = np.zeros(3, dtype=np.float32)
        if plane == "xy":
            corr_xyz[0] = -shift[0]
            corr_xyz[1] = -shift[1]
        elif plane == "xz":
            corr_xyz[0] = -shift[0]
            corr_xyz[2] = shift[1]
        else:  # yz
            corr_xyz[1] = -shift[0]
            corr_xyz[2] = -shift[1]

        max_corr = max(1.0, radius_px * 0.8)  
        corr_norm = float(np.linalg.norm(corr_xyz))
        if corr_norm > max_corr:
            corr_xyz *= (max_corr / corr_norm)
        # 相关性变差时降低位移校正幅度，减少噪声驱动的抖动。
        if error > 0.7:
            corr_xyz *= 0.20
        elif error > 0.5:
            corr_xyz *= 0.50

        next_center = pred + corr_xyz
        # 轴向保持严格步进；guide 仅作用在切面内，避免横向漂移。
        blend = float(np.clip(guide_weight, 0.0, 0.35))
        if plane == "xy":
            next_center[0] = (1.0 - blend) * next_center[0] + blend * guide[0]
            next_center[1] = (1.0 - blend) * next_center[1] + blend * guide[1]
            next_center[2] = pred[2]
        elif plane == "xz":
            next_center[0] = (1.0 - blend) * next_center[0] + blend * guide[0]
            next_center[2] = (1.0 - blend) * next_center[2] + blend * guide[2]
            next_center[1] = pred[1]
        else:  # yz
            next_center[1] = (1.0 - blend) * next_center[1] + blend * guide[1]
            next_center[2] = (1.0 - blend) * next_center[2] + blend * guide[2]
            next_center[0] = pred[0]
        next_center = np.clip(next_center, [0, 0, 0], [vol_zyx.shape[2] - 1, vol_zyx.shape[1] - 1, vol_zyx.shape[0] - 1])

        centers.append(next_center.astype(np.float32))
        center = next_center

    # 末端强制回到用户给定终点，保证输出曲线精确经过 start/end。
    centers.append(end_xyz.copy())
    pts = np.asarray(centers, dtype=np.float32)
    return smooth_curve(pts, start_xyz, end_xyz, curve_points=curve_points, curvature=curvature)


def smooth_curve(points_xyz: np.ndarray, start_xyz: np.ndarray, end_xyz: np.ndarray, curve_points: int, curvature: float) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    curve_points = max(2, int(curve_points))
    curvature = float(np.clip(curvature, 0.0, 1.0))

    if len(points_xyz) < 4:
        t = np.linspace(0.0, 1.0, curve_points, dtype=np.float32)
        curve = start_xyz[None, :] * (1.0 - t[:, None]) + end_xyz[None, :] * t[:, None]
        curve[0], curve[-1] = start_xyz, end_xyz
        return curve.astype(np.float32)

    d = np.linalg.norm(np.diff(points_xyz, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-6:
        t = np.linspace(0.0, 1.0, curve_points, dtype=np.float32)
        curve = start_xyz[None, :] * (1.0 - t[:, None]) + end_xyz[None, :] * t[:, None]
        curve[0], curve[-1] = start_xyz, end_xyz
        return curve.astype(np.float32)
    
    u = s / s[-1]
    smooth_factor = (1.0 - curvature) * len(points_xyz) * 0.75
    try:
        tck, _ = splprep([points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]], u=u, k=min(3, len(points_xyz) - 1), s=smooth_factor)
        uq = np.linspace(0.0, 1.0, curve_points, dtype=np.float32)
        x, y, z = splev(uq, tck)
        curve = np.vstack([x, y, z]).T.astype(np.float32)
    except Exception:
        uq = np.linspace(0.0, 1.0, curve_points, dtype=np.float32)
        curve = np.empty((curve_points, 3), dtype=np.float32)
        for i in range(3):
            curve[:, i] = np.interp(uq, u, points_xyz[:, i])

    curve[0], curve[-1] = start_xyz, end_xyz
    return curve.astype(np.float32)


def resample_curve_by_spacing(curve_xyz: np.ndarray, spacing_px: float) -> np.ndarray:
    curve_xyz = np.asarray(curve_xyz, dtype=np.float32)
    if len(curve_xyz) <= 1:
        return curve_xyz.copy()
    spacing_px = max(float(spacing_px), 1e-3)
    seg = np.linalg.norm(np.diff(curve_xyz, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= spacing_px:
        return np.vstack([curve_xyz[0], curve_xyz[-1]]).astype(np.float32)
    q = np.arange(0.0, total + 0.5 * spacing_px, spacing_px, dtype=np.float32)
    q[-1] = total
    out = np.empty((len(q), 3), dtype=np.float32)
    for i in range(3):
        out[:, i] = np.interp(q, s, curve_xyz[:, i])
    return out


def interpolate_angles(t: np.ndarray, start_ang: np.ndarray, end_ang: np.ndarray) -> np.ndarray:
    return start_ang + t[:, None] * (end_ang - start_ang)


def _curve_to_der_points(curve: np.ndarray) -> np.ndarray:
    n = len(curve)
    tangents = np.zeros_like(curve)
    if n == 1:
        tangents[0] = 1.0
        return tangents.T
    tangents[0] = curve[1] - curve[0]
    tangents[-1] = curve[-1] - curve[-2]
    if n > 2:
        tangents[1:-1] = 0.5 * (curve[2:] - curve[:-2])
    return tangents.T


def write_axm(path: Path, curve_xyz_px: np.ndarray, voxel_size: float) -> Path:
    """
    写入 ArtiaX CurvedLine 格式的 .axm 文件，可在 ChimeraX+ArtiaX 中打开为曲线模型。
    格式与 ArtiaX CurvedLine.write_file 一致（内部为 npz）：model_type, particle_pos, degree,
    smooth, resolution, points (3×n), der_points (3×n)。通过 open(..., 'wb') 写入，保证文件后缀为 .axm
    而非 .npz。在 ArtiaX 中通过 Geometric Models → Open Geomodel... 选择本文件打开。
    | 参数名称               | 含义说明                                         |
    | :----------------- | :------------------------------------------- |
    | **`model_type`**   | 必须为 `"CurvedLine"`，ArtiaX 据此识别模型类型并调用相应的渲染类。 |
    | **`particle_pos`** | 原始颗粒坐标列表。在 ArtiaX 中，这些点用于定义曲线的控制点。           |
    | **`degree`**       | 样条曲线的阶数（通常为 3），决定了曲线的平滑程度。                   |
    | **`smooth`**       | 样条拟合的平滑参数。`0` 表示曲线必须经过所有 `particle_pos` 点。   |
    | **`resolution`**   | 渲染分辨率，即在控制点之间插值生成的显示点数量。                     |
    | **`points`**       | 预计算的曲线采样点坐标，形状为 `(3, n)`。ArtiaX 渲染时直接使用这些点。  |
    | **`der_points`**   | 曲线各点处的导数（切向量），形状为 `(3, n)`，用于计算颗粒的取向。        |
    参考: https://github.com/FrangakisLab/ArtiaX/blob/main/src/geometricmodel/CurvedLine.py
    """
    path = Path(path).with_suffix(".axm")
    curve_a = np.asarray(curve_xyz_px, dtype=np.float32) * float(voxel_size)
    if len(curve_a) < 2:
        curve_a = np.vstack([curve_a[0], curve_a[0]])
    # curve_a = curve_a[:40]
    # print(curve_a)
    with open(path, "wb") as f:
        np.savez(
            f,
            model_type="CurvedLine",
            particle_pos=curve_a,
            degree=3,
            smooth=0.0,
            resolution=int(len(curve_a)),
            points=curve_a.T,
            der_points=_curve_to_der_points(curve_a),
        )
    return path


def write_star(out_path: Path, coords: np.ndarray, angles: np.ndarray, origins: Optional[np.ndarray] = None) -> None:
    if starfile is None:
        raise ImportError("please install starfile: pip install starfile")
    import pandas as pd

    if origins is None:
        origins = np.zeros((len(coords), 3), dtype=np.float32)
    df = pd.DataFrame(
        {
            "rlnCoordinateX": coords[:, 0],
            "rlnCoordinateY": coords[:, 1],
            "rlnCoordinateZ": coords[:, 2],
            "rlnOriginX": origins[:, 0],
            "rlnOriginY": origins[:, 1],
            "rlnOriginZ": origins[:, 2],
            "rlnAngleRot": angles[:, 0],
            "rlnAngleTilt": angles[:, 1],
            "rlnAnglePsi": angles[:, 2],
        }
    )
    starfile.write({"0": df}, out_path, overwrite=True)


_worker_vol: Optional[np.ndarray] = None
_worker_shm = None
_worker_fibers: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None
_worker_track_kw: Dict[str, object] = {}


def _worker_cleanup() -> None:
    global _worker_shm
    if _worker_shm is not None:
        try:
            _worker_shm.close()
        except Exception:
            pass
        _worker_shm = None


def _init_worker(shm_name: str, shape: Tuple[int, int, int], dtype_str: str, fibers: List, track_kw: Dict[str, object]) -> None:
    global _worker_vol, _worker_shm, _worker_fibers, _worker_track_kw
    _worker_shm = shared_memory.SharedMemory(name=shm_name)
    _worker_vol = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_worker_shm.buf)
    _worker_fibers = fibers
    _worker_track_kw = track_kw
    atexit.register(_worker_cleanup)


def _trace_worker(idx: int) -> Tuple[int, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    # worker 仅接收索引，体积通过共享内存读取，避免重复拷贝大数组。
    s, e, a0, a1 = _worker_fibers[idx]
    curve = trace_fiber_crosscorr(_worker_vol, s, e, **_worker_track_kw)
    return idx, curve.astype(np.float32), (s, e, a0, a1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace fibers in .rec tomogram by 2D slice cross-correlation and export .axm/.star.")
    parser.add_argument("--star", required=True, help="fiber start/end .star path")
    parser.add_argument("--rec", required=True, help="input .rec tomogram path")
    parser.add_argument("--voxel-size", required=True, type=float, help="voxel size in Angstrom")

    parser.add_argument("--radius", type=float, default=250.0, help="fiber radius in Angstrom")
    parser.add_argument("--spacing", type=float, default=40.0, help="particle spacing along curve in Angstrom")
    parser.add_argument("--step", type=int, default=2, help="merge +/-step neighboring slices (2*step+1 total) before CC, default 2")

    parser.add_argument("--curve-points", type=int, default=80, help="curve sample points per fiber")
    parser.add_argument("--curvature", type=float, default=0.25, help="curve smoothing control [0,1], higher follows data more")
    parser.add_argument("--margin-ratio", type=float, default=0.75, help="extra sampling margin in radius units (default 0.75)")
    parser.add_argument("--guide-weight", type=float, default=0.01, help="linear start->end guide weight during tracking (default 0.01, [0.0, 0.35])")
    parser.add_argument("--cc-upsample", type=int, default=10, help="phase cross-correlation upsample factor (default 10)")
    parser.add_argument("--gaussian-sigma", type=float, default=2.0, help="3D gaussian sigma for enhancement (default 2.0)")
    parser.add_argument("--closing-size", type=int, default=3, help="3D grey-closing kernel size, >1 improves continuity (default 3)")

    parser.add_argument("--fiber-index", default=None, help="fiber indices: e.g. 0,2,5 or 0-3")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4)), help="parallel workers")
    parser.add_argument("--debug", action="store_true", help="save intermediate MRC files in rec folder")
    args = parser.parse_args()

    if args.voxel_size <= 0:
        print("error: --voxel-size must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.radius <= 0:
        print("error: --radius must be > 0", file=sys.stderr)
        sys.exit(1)

    star_path = Path(args.star).resolve()
    rec_path = Path(args.rec).resolve()
    if not star_path.exists():
        print(f"error: star file not found: {star_path}", file=sys.stderr)
        sys.exit(1)
    if not rec_path.exists():
        print(f"error: rec file not found: {rec_path}", file=sys.stderr)
        sys.exit(1)

    fibers, has_orphan = read_star_fibers(star_path)
    if has_orphan:
        print("warning: star row count is odd, last unmatched start point is ignored.", file=sys.stderr)
    if not fibers:
        print("error: no valid start/end fiber pairs found.", file=sys.stderr)
        sys.exit(1)

    selected = parse_fiber_index_list(args.fiber_index, len(fibers))
    if not selected:
        print("error: no fiber selected.", file=sys.stderr)
        sys.exit(1)

    print(f"loading volume: {rec_path}")
    vol = load_volume(rec_path)
    print(f"volume shape (z,y,x): {vol.shape}")

    print("enhancing volume ...")
    voxel = float(args.voxel_size)
    radius_px = float(args.radius) / voxel
    spacing_px = float(args.spacing) / voxel
    rec_dir = rec_path.parent
    debug_mrc_prefix = rec_dir / rec_path.stem if args.debug else None
    vol_enh = enhance_volume(
        vol=vol,
        voxel_size=voxel,
        gaussian_sigma=float(args.gaussian_sigma),
        closing_size=int(args.closing_size),
        debug_mrc_prefix=debug_mrc_prefix,
    )

    # 追踪参数统一打包传入 worker，避免多进程时参数散落。
    track_kw = dict(
        radius_px=radius_px,
        merge_step=int(args.step),
        curve_points=int(args.curve_points),
        curvature=float(args.curvature),
        margin_ratio=float(args.margin_ratio),
        guide_weight=float(args.guide_weight),
        cc_upsample=int(args.cc_upsample),
        debug_mrc_prefix=debug_mrc_prefix,
        voxel_size=voxel,
    )

    n_workers = max(1, min(int(args.workers), len(selected)))
    results: Dict[int, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}

    # 将增强后体积放入共享内存，子进程只读访问。
    shm = shared_memory.SharedMemory(create=True, size=vol_enh.nbytes)
    try:
        shm_arr = np.ndarray(vol_enh.shape, dtype=vol_enh.dtype, buffer=shm.buf)
        shm_arr[:] = vol_enh[:]
        del shm_arr

        if n_workers > 1:
            print(f"tracing fibers in parallel, workers={n_workers}")
            with multiprocessing.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(shm.name, vol_enh.shape, str(vol_enh.dtype), fibers, track_kw),
            ) as pool:
                for idx, curve, meta in pool.imap_unordered(_trace_worker, selected):
                    results[idx] = (curve, meta)
                    print(f"  fiber {idx}: {len(curve)} curve points")
        else:
            print("tracing fibers in single process")
            _init_worker(shm.name, vol_enh.shape, str(vol_enh.dtype), fibers, track_kw)
            for idx in selected:
                ridx, curve, meta = _trace_worker(idx)
                results[ridx] = (curve, meta)
                print(f"  fiber {ridx}: {len(curve)} curve points")
            _worker_cleanup()
    finally:
        shm.close()
        shm.unlink()

    # 输出每条纤维的可视化曲线（ArtiaX .axm）
    axm_paths = []
    for idx in selected:
        curve, _ = results[idx]
        axm_path = rec_dir / f"{rec_path.stem}_fiber{idx}.axm"
        write_axm(axm_path, curve, voxel_size=voxel)
        axm_paths.append(axm_path)
    print(f"saved {len(axm_paths)} AXM files in {rec_dir}")

    # 按 spacing 沿曲线重采样并写颗粒 STAR
    out_star = rec_dir / f"{rec_path.stem}_particles.star"
    all_coords = []
    all_angles = []
    for idx in selected:
        curve, (_, _, a0, a1) = results[idx]
        pts = resample_curve_by_spacing(curve, spacing_px=spacing_px)
        if len(pts) == 1:
            t = np.array([0.0], dtype=np.float32)
        else:
            t = np.linspace(0.0, 1.0, len(pts), dtype=np.float32)
        ang = interpolate_angles(t, a0, a1).astype(np.float32)
        all_coords.append(pts)
        all_angles.append(ang)

    coords = np.vstack(all_coords).astype(np.float32)
    angles = np.vstack(all_angles).astype(np.float32)
    write_star(out_star, coords, angles)
    print(f"STAR saved: {out_star}, total particles: {len(coords)}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
