#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 SimpleITK + cucim/cupy，对 tomogram 中的管状结构进行增强与骨架提取，
再结合 96^3 模板做局部模板匹配，自动筛选疑似 microtubule doublet 骨架，
输出：
- 每条通过筛选的骨架曲线的 .axm（ArtiaX CurvedLine，用于 ChimeraX 可视化）
- 沿骨架按指定间距采样的粒子 .star（位置 + 近似取向）
- （可选 --debug）输出 .cmm marker 文件，记录模板匹配采样点及得分等，用于在 ChimeraX 中调参与排错。

依赖：
- 必需：numpy, scipy, SimpleITK, mrcfile, starfile, pandas
- 可选：cupy, cucim，用于在 GPU 上加速 Frangi vesselness 与 NCC 计算
"""
from __future__ import annotations

import argparse
import atexit
import math
import multiprocessing
import os
import mrcfile
import starfile
from multiprocessing import shared_memory
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import SimpleITK as sitk
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation as R

import cupy as cp
from cucim import skimage as cskimage
from cucim.core.operations import intensity as cintensity

# ==============================
# 基础工具函数
# ==============================

def load_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True, mode="r") as m:
        vol = np.asarray(m.data, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape={vol.shape}")
    return vol


def save_mrc(path: Path, data: np.ndarray, voxel_size: float = 1.0) -> None:
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(np.asarray(data, dtype=np.float32))
        try:
            m.voxel_size = float(voxel_size)
        except Exception:
            print(f"Failed to set voxel size, but saving anyway.")
            pass


def resample_curve_by_spacing(curve_xyz: np.ndarray, spacing_px: float) -> np.ndarray:
    """
    沿曲线按固定弧长间隔重采样（像素单位），返回新的点序列。
    """
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


def smooth_curve(points_xyz: np.ndarray, curve_points: int = 200, curvature: float = 0.25) -> np.ndarray:
    """
    对离散点序列做一维样条平滑，输出更平滑的曲线。
    curvature ∈ [0,1]：0 更直，1 更贴合原始点。
    """
    pts = np.asarray(points_xyz, dtype=np.float32)
    n = len(pts)
    if n < 2:
        return pts.copy()
    curve_points = max(2, int(curve_points))
    # 弧长参数
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total < 1e-6:
        # 退化为首末两点直线
        t = np.linspace(0.0, 1.0, curve_points, dtype=np.float32)
        out = pts[0][None, :] * (1.0 - t[:, None]) + pts[-1][None, :] * t[:, None]
        return out.astype(np.float32)

    curv = float(np.clip(curvature, 0.0, 1.0))
    smooth_factor = (1.0 - curv) * n * 0.75

    try:
        us = s / total
        tck_x = UnivariateSpline(us, pts[:, 0], s=smooth_factor, k=min(3, n - 1))
        tck_y = UnivariateSpline(us, pts[:, 1], s=smooth_factor, k=min(3, n - 1))
        tck_z = UnivariateSpline(us, pts[:, 2], s=smooth_factor, k=min(3, n - 1))
        uq = np.linspace(0.0, 1.0, curve_points, dtype=np.float32)
        x = tck_x(uq)
        y = tck_y(uq)
        z = tck_z(uq)
        curve = np.vstack([x, y, z]).T.astype(np.float32)
    except Exception:
        # 出错时退化为线性插值
        uq = np.linspace(0.0, 1.0, curve_points, dtype=np.float32)
        curve = np.empty((curve_points, 3), dtype=np.float32)
        for i in range(3):
            curve[:, i] = np.interp(uq, s / total, pts[:, i])

    curve[0] = pts[0]
    curve[-1] = pts[-1]
    return curve.astype(np.float32)


def _curve_to_der_points(curve: np.ndarray) -> np.ndarray:
    """
    由曲线点 (n,3) 计算各点处切向量，返回 (3,n)（ArtiaX CurvedLine 所需）。
    """
    curve = np.asarray(curve, dtype=np.float32)
    n = len(curve)
    tangents = np.zeros_like(curve)
    if n == 1:
        tangents[0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return tangents.T
    tangents[0] = curve[1] - curve[0]
    tangents[-1] = curve[-1] - curve[-2]
    if n > 2:
        tangents[1:-1] = 0.5 * (curve[2:] - curve[:-2])
    return tangents.T.astype(np.float32)


def write_axm(path: Path, curve_xyz_px: np.ndarray, voxel_size: float) -> Path:
    """
    写 ArtiaX CurvedLine .axm 文件（内部为 npz），坐标单位为 Å。
    """
    import numpy as _np

    path = Path(path).with_suffix(".axm")
    curve = _np.asarray(curve_xyz_px, dtype=_np.float32) * float(voxel_size)
    if len(curve) < 2:
        curve = _np.vstack([curve[0], curve[0]])
    points = curve.T
    der_points = _curve_to_der_points(curve)
    particle_pos = curve
    with open(path, "wb") as f:
        _np.savez(
            f,
            model_type="CurvedLine",
            particle_pos=particle_pos,
            degree=3,
            smooth=0.0,
            resolution=int(len(curve)),
            points=points,
            der_points=der_points,
        )
    return path


def write_star(out_path: Path, coords: np.ndarray, angles: np.ndarray) -> None:
    """
    将粒子坐标与欧拉角写入 STAR 文件。
    坐标单位：像素；角度单位：度；origin 填 0。
    """
    import pandas as pd

    coords = np.asarray(coords, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float32)
    origins = np.zeros_like(coords, dtype=np.float32)
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


def vector_to_euler_zyz(vec: np.ndarray) -> Tuple[float, float, float]:
    """
    将 3D 单位向量映射为 ZYZ 欧拉角 (rot, tilt, psi)，
    约定为：找到将 z 轴 [0,0,1] 旋转到 vec 的旋转矩阵，再转为 ZYZ。
    """
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-6:
        return 0.0, 0.0, 0.0
    v = v / n
    try:
        # R * [0,0,1] ≈ v
        rot, _ = R.align_vectors([v], [np.array([0.0, 0.0, 1.0])])
        ang = rot.as_euler("ZYZ", degrees=True)[0]
        return float(ang[0]), float(ang[1]), float(ang[2])
    except Exception:
        # 简单 fallback：从笛卡尔到球坐标
        # z = cos(tilt)，xy 投影决定 rot，psi=0
        z = float(v[2])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, z))))
        rot = math.degrees(math.atan2(float(v[1]), float(v[0])))
        return rot, tilt, 0.0


# ==============================
# Vesselness + Skeleton
# ==============================

def normalize_percentile(vol: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """使用 GPU 计算分位数 + 归一化到 [0,1]。"""
    v_gpu = cp.asarray(vol, dtype=cp.float32)
    lo_gpu, hi_gpu = cp.percentile(v_gpu, [low, high])
    lo = float(lo_gpu)
    hi = float(hi_gpu)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    norm_gpu = cintensity.normalize_data(v_gpu, 1.0, lo, hi, type="range")
    norm_gpu = cp.clip(norm_gpu, 0.0, 1.0)
    return cp.asnumpy(norm_gpu).astype(np.float32)


def gaussian_smooth(vol: np.ndarray, sigma: float) -> np.ndarray:
    """使用 GPU 做 3D 高斯平滑。"""
    v_gpu = cp.asarray(vol, dtype=cp.float32)
    gauss_sigma = max(0.5, float(sigma))
    v_gpu = cskimage.filters.gaussian(v_gpu, sigma=gauss_sigma, preserve_range=True)
    return cp.asnumpy(v_gpu).astype(np.float32)


def sato_vesselness(
    vol_zyx: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    sigma_steps: int,
) -> np.ndarray:
    """
    使用 Sato 管状增强（Hessian-based tubularity）对 3D 体积做多尺度增强。

    参数说明（所有 sigma 以“像素”为单位）：
        vol_zyx    : 归一化 / 可选 invert 后的体积，形状 (z, y, x)
        sigma_min  : Sato 多尺度中最小 sigma（像素），建议略小于目标内半径
        sigma_max  : 最大 sigma（像素），推荐略小于或接近目标外半径
        sigma_steps: sigma 采样个数，通常 3–6 即可
        bright_tubes: True 表示“亮管”增强；若原图是暗管，先 --invert
    返回值：
        vesselness 体积，与输入同 shape，已在 [0,1] 归一化。
    """
    v = np.asarray(vol_zyx, dtype=np.float32)
    sigma_min = max(float(sigma_min), 1)
    sigma_max = max(float(sigma_max), sigma_min)
    sigma_steps = max(int(sigma_steps), 1)
    sigmas = np.linspace(sigma_min, sigma_max, sigma_steps, dtype=np.float32)
    sigmas = np.unique(sigmas).tolist()
    if not sigmas:
        sigmas = [sigma_min]

    # 使用 GPU 版 cucim：需要 cupy + cucim，输入必须是 cupy.ndarray
    v_gpu = cp.asarray(v, dtype=cp.float32)
    out_gpu = cskimage.filters.sato(v_gpu, sigmas=sigmas, black_ridges=True, mode="reflect",)
    out = cp.asnumpy(out_gpu).astype(np.float32)
    out[np.isnan(out)] = 0.0
    out[np.isinf(out)] = 0.0
    if out.size > 0 and float(out.max()) > 0:
        out /= float(out.max())
    return out.astype(np.float32)


def extract_skeletons(
    vesselness_zyx: np.ndarray,
    vesselness_percentile: float,
    min_region_voxels: int,
    voxel_size: float,
    min_skel_length_A: float,
    debug_prefix: Optional[Path] = None,
) -> List[np.ndarray]:
    """
    从 vesselness 体积中构建候选管状骨架：

    1) 对 vesselness 做百分位阈值二值化，得到初始 mask；
    2) 在 GPU 上使用 cucim.morphology.remove_small_holes / remove_small_objects 清理小孔与小连通域（基于体素数）；
    3) 将清理后的 mask 转回 CPU，用 SimpleITK.BinaryThinning 做 3D 骨架提取；
    4) 对骨架连通域再次分割，并按 PCA 主轴排序为近似有序曲线；
    5) 以物理长度（min_skel_length_A）过滤过短骨架。

    返回：每条骨架的一组点 (n_i,3)，坐标单位为像素 (x,y,z)，已排序但尚未平滑。
    """
    v = np.asarray(vesselness_zyx, dtype=np.float32)

    # 1) 在 GPU 上做百分位阈值二值化
    v_gpu = cp.asarray(v, dtype=cp.float32)
    finite = cp.isfinite(v_gpu)
    if not bool(finite.sum().get()):
        print("  vesselness has no finite values")
        return []
    th = float(cp.percentile(v_gpu[finite], vesselness_percentile))
    mask_gpu = v_gpu >= th
    if not bool(mask_gpu.sum().get()):
        print("  mask is empty after thresholding")
        return []
    if debug_prefix is not None:
        save_mrc(debug_prefix.with_name(debug_prefix.name + "_4_mask_raw.mrc"), cp.asnumpy(mask_gpu.astype(cp.float32)), voxel_size=voxel_size)

    # 2) 删除小前景（1）
    mask_gpu = cskimage.morphology.remove_small_objects(mask_gpu, min_size=max(1, min_region_voxels), connectivity=1)
    if not bool(mask_gpu.sum().get()):
        print("  mask removed completely after small-object filtering")
        return []
    if debug_prefix is not None:
        save_mrc(debug_prefix.with_name(debug_prefix.name + "_5_mask_clean_objects.mrc"), cp.asnumpy(mask_gpu.astype(cp.float32)), voxel_size=voxel_size)
    
    # 2）删除小背景（0）
    mask_gpu = cskimage.morphology.remove_small_holes(mask_gpu, area_threshold=max(1, min_region_voxels), connectivity=1)  
    if not bool(mask_gpu.sum().get()):
        print("  mask removed completely after small-hole filtering")
        return []
    if debug_prefix is not None:
        save_mrc(debug_prefix.with_name(debug_prefix.name + "_6_mask_clean_holes.mrc"), cp.asnumpy(mask_gpu.astype(cp.float32)), voxel_size=voxel_size)

    # 3) 3D 骨架
    mask = cp.asnumpy(mask_gpu.astype(np.uint8))
    mask_img = sitk.GetImageFromArray(mask)
    skel_img = sitk.BinaryThinning(mask_img)
    skel_np = sitk.GetArrayFromImage(skel_img)
    if skel_np.sum() == 0:
        print("  skeleton is empty after thinning")
        return []
    if debug_prefix is not None:
        save_mrc(debug_prefix.with_name(debug_prefix.name + "_7_skel.mrc"), skel_np.astype(np.float32), voxel_size=voxel_size)

    # 4) 骨架连通域，再按 PCA 排序得到“曲线”
    skel_cc = sitk.ConnectedComponent(skel_img)
    skel_cc_np = sitk.GetArrayFromImage(skel_cc)
    ls2 = sitk.LabelShapeStatisticsImageFilter()
    ls2.Execute(skel_cc)

    curves: List[np.ndarray] = []
    for lab in ls2.GetLabels():
        # 当前骨架连通域所有点
        pts_zyx = np.argwhere(skel_cc_np == int(lab))
        if pts_zyx.size == 0:
            continue
        pts_xyz = np.stack(
            [pts_zyx[:, 2].astype(np.float32), pts_zyx[:, 1].astype(np.float32), pts_zyx[:, 0].astype(np.float32)],
            axis=1,
        )
        if len(pts_xyz) < 2:
            continue

        # PCA 主轴
        mean = pts_xyz.mean(axis=0)
        centered = pts_xyz - mean
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            main_dir = vh[0]
        except Exception:
            print("  PCA failed, skipping curve")
            continue

        proj = centered @ main_dir
        order = np.argsort(proj)
        pts_sorted = pts_xyz[order]

        # 长度过滤（沿主轴差值近似）
        length_vox = float(proj.max() - proj.min())
        if length_vox * float(voxel_size) < min_skel_length_A:
            continue

        # 邻近去重，避免点过密
        dedup: List[int] = [0]
        for i in range(1, len(pts_sorted)):
            if np.linalg.norm(pts_sorted[i] - pts_sorted[dedup[-1]]) >= 1.0:
                dedup.append(i)
        if dedup[-1] != len(pts_sorted) - 1:
            dedup.append(len(pts_sorted) - 1)
        pts_dedup = pts_sorted[dedup]
        if len(pts_dedup) < 2:
            continue

        curves.append(pts_dedup.astype(np.float32))

    return curves


# ==============================
# 模板匹配（NCC） + 多进程 / GPU
# ==============================

def ncc_score(patch: np.ndarray, template: np.ndarray, use_gpu: bool = False) -> float:
    """
    计算 patch 与 template 的归一化互相关 (NCC)，返回 [-1,1]。
    若 use_gpu=True 且 cupy 可用，则在 GPU 上计算；否则在 CPU 上计算。
    """
    p = np.asarray(patch, dtype=np.float32)
    t = np.asarray(template, dtype=np.float32)
    if p.shape != t.shape:
        return 0.0

    if use_gpu:
        try:
            pg = cp.asarray(p, dtype=cp.float32)
            tg = cp.asarray(t, dtype=cp.float32)
            pg = pg - pg.mean()
            tg = tg - tg.mean()
            num = float((pg * tg).sum())
            den = float(cp.sqrt((pg * pg).sum() * (tg * tg).sum()))
            if den <= 0 or not math.isfinite(den):
                return 0.0
            s = num / den
            if not math.isfinite(s):
                return 0.0
            return float(max(-1.0, min(1.0, s)))
        except Exception:
            # 任何异常时退回 CPU
            pass

    # CPU 版
    p0 = p - float(p.mean())
    t0 = t - float(t.mean())
    num = float((p0 * t0).sum())
    den = float(math.sqrt(float((p0 * p0).sum()) * float((t0 * t0).sum())))
    if den <= 0 or not math.isfinite(den):
        return 0.0
    s = num / den
    if not math.isfinite(s):
        return 0.0
    return float(max(-1.0, min(1.0, s)))


_worker_vol: Optional[np.ndarray] = None
_worker_shm = None
_worker_template: Optional[np.ndarray] = None
_worker_cfg: Dict[str, object] = {}


def _worker_cleanup() -> None:
    global _worker_shm
    if _worker_shm is not None:
        try:
            _worker_shm.close()
        except Exception:
            pass
        _worker_shm = None


def _init_match_worker(
    shm_name: str,
    shape: Tuple[int, int, int],
    dtype_str: str,
    template: np.ndarray,
    cfg: Dict[str, object],
) -> None:
    """
    多进程 worker 初始化：从共享内存中映射 tomogram 体积，将模板与配置注入全局。
    """
    global _worker_vol, _worker_shm, _worker_template, _worker_cfg
    _worker_shm = shared_memory.SharedMemory(name=shm_name)
    _worker_vol = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_worker_shm.buf)
    _worker_template = np.asarray(template, dtype=np.float32)
    _worker_cfg = dict(cfg)
    atexit.register(_worker_cleanup)


def _process_curve_worker(args: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    对单条骨架曲线：
    - 按 block_step_px 间距重采样中心点
    - 在共享 tomogram 上裁剪 patch，与模板做 NCC
    返回：
        (curve_idx, curve_xyz, centers_xyz, scores)
    """
    idx, curve = args
    vol = _worker_vol
    template = _worker_template
    cfg = _worker_cfg
    if vol is None or template is None:
        raise RuntimeError("worker volume/template not initialized")

    use_gpu = bool(cfg.get("use_gpu", False))
    block_step_px = float(cfg.get("block_step_px", 8.0))

    curve = np.asarray(curve, dtype=np.float32)
    if len(curve) < 2:
        return idx, curve, np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    centers = resample_curve_by_spacing(curve, spacing_px=block_step_px)
    if len(centers) == 0:
        return idx, curve, np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    tz, ty, tx = template.shape  # (z,y,x)
    hz, hy, hx = tz // 2, ty // 2, tx // 2
    sz, sy, sx = vol.shape

    valid_centers: List[np.ndarray] = []
    scores: List[float] = []

    for c in centers:
        cx, cy, cz = np.round(c).astype(int)
        z0 = cz - hz
        z1 = z0 + tz
        y0 = cy - hy
        y1 = y0 + ty
        x0 = cx - hx
        x1 = x0 + tx
        if z0 < 0 or y0 < 0 or x0 < 0 or z1 > sz or y1 > sy or x1 > sx:
            continue
        patch = vol[z0:z1, y0:y1, x0:x1]
        if patch.shape != template.shape:
            continue
        s = ncc_score(patch, template, use_gpu=use_gpu)
        valid_centers.append(c.astype(np.float32))
        scores.append(float(s))

    if not valid_centers:
        return idx, curve, np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    centers_a = np.vstack(valid_centers).astype(np.float32)
    scores_a = np.asarray(scores, dtype=np.float32)
    return idx, curve, centers_a, scores_a


# ==============================
# CMM debug 输出
# ==============================

def write_cmm_markers(
    path: Path,
    markers: Sequence[Tuple[int, np.ndarray, float, bool]],
) -> None:
    """
    写 ChimeraX .cmm marker 文件。
    markers: 序列 (fiber_idx, center_xyz_px, ncc_score, accepted_bool)
    坐标单位：像素 (x,y,z)。可在 ChimeraX 中与原始 tomogram 对齐观察。
    """
    if not markers:
        return
    # 统计得分范围，用于颜色映射
    scores = np.asarray([m[2] for m in markers], dtype=np.float32)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=np.float32)
    s_min = float(finite.min())
    s_max = float(finite.max())
    if not math.isfinite(s_min):
        s_min = 0.0
    if not math.isfinite(s_max):
        s_max = 1.0
    if abs(s_max - s_min) < 1e-6:
        s_min, s_max = 0.0, 1.0

    def _color_from_score(s: float, accepted: bool) -> Tuple[float, float, float]:
        # 归一化到 [0,1]
        sn = (s - s_min) / (s_max - s_min)
        sn = float(max(0.0, min(1.0, sn)))
        # 简单蓝→红：低分蓝，高分红
        r = sn
        g = 0.2 + 0.6 * sn
        b = 1.0 - sn
        if not accepted:
            # 未被接受的骨架点调暗
            r *= 0.5
            g *= 0.5
            b *= 0.5
        return r, g, b

    lines: List[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<marker_set name="fiber2star_v3_debug">')
    mid = 1
    for fiber_idx, center_xyz, score, accepted in markers:
        x, y, z = map(float, center_xyz.tolist())
        r, g, b = _color_from_score(score, accepted)
        radius = 0.5 + 1.5 * max(0.0, min(1.0, (score - s_min) / (s_max - s_min)))
        note = f"fiber={fiber_idx};score={score:.4f};accepted={int(accepted)}"
        lines.append(
            f'<marker id="{mid}" x="{x:.3f}" y="{y:.3f}" z="{z:.3f}" '
            f'r="{r:.3f}" g="{g:.3f}" b="{b:.3f}" radius="{radius:.3f}" note="{note}"/>'
        )
        mid += 1
    lines.append("</marker_set>")
    path.write_text("\n".join(lines), encoding="utf-8")


# ==============================
# 主流程
# ==============================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tomogram + template → SimpleITK vesselness + skeleton → block-wise NCC template matching → "
            "doublet-like curves → ArtiaX .axm + particles .star + optional ChimeraX .cmm debug markers"
        )
    )
    parser.add_argument("--tomo", required=True, help="input tomogram .mrc path")
    parser.add_argument("--template", required=True, help="template .mrc path (e.g. TZ_template_bin4.mrc)")
    parser.add_argument("--voxel-size", type=float, required=True, help="voxel size (Å) (required, for coordinate scale conversion)")
    parser.add_argument("--bin", type=int, default=2, help="integer binning factor applied to both tomogram and template before processing (default 2)")

    # vesselness / skeleton 参数
    parser.add_argument("--gauss-sigma", type=float, default=1.0, help="3D gaussian sigma for vesselness enhancement (default 1.0)")
    parser.add_argument("--sigma-min", type=float, default=11.0, help="minimum sigma (voxel) for multi-scale vesselness (default 11.0), approximately 0.5 * small radius of the filament")
    parser.add_argument("--sigma-max", type=float, default=13.0, help="maximum sigma (voxel) for multi-scale vesselness (default 13.0), approximately 1.0 * large radius of the filament")
    parser.add_argument("--sigma-steps", type=int, default=4, help="number of scales for vesselness (default 4)")
    parser.add_argument("--vesselness-percentile", type=float, default=80.0, help="percentile threshold on vesselness to build binary mask (default 80)")
    parser.add_argument("--min-region-voxels", type=int, default=200000, help="minimum voxel count for connected components before skeletonization (default 200000)")
    parser.add_argument("--min-skel-length", type=float, default=200.0, help="minimum skeleton length in Angstrom (approximate, default 200)")

    # 模板匹配 / 采样
    parser.add_argument("--block-step", type=float, default=80.0, help="sampling step along skeleton for template matching (Angstrom, default 80)")
    parser.add_argument("--match-threshold", type=float, default=0.2, help="NCC threshold for accepting a curve (mean score or good fraction) (default 0.2)")
    parser.add_argument("--good-fraction", type=float, default=0.3, help="minimum fraction of sampling points with NCC >= match-threshold to accept a curve (default 0.3)")
    parser.add_argument("--spacing", type=float, default=40.0, help="particle spacing along accepted curves (Angstrom, default 40)")

    # 执行模式
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4)), help="number of workers for curve-level NCC (CPU only; GPU mode uses single process)")
    parser.add_argument("--out-prefix", default=None, help="output file prefix (default: tomo stem in its folder)")
    parser.add_argument("--debug", action="store_true", help="enable debug outputs (some intermediate MRCs and a .cmm marker file)")

    args = parser.parse_args()

    voxel = float(args.voxel_size)
    if args.voxel_size <= 0:
        print("error: --voxel-size must be > 0")
        return

    tomo_path = Path(args.tomo).resolve()
    tpl_path = Path(args.template).resolve()
    if not tomo_path.exists():
        raise SystemExit(f"error: tomogram not found: {tomo_path}")
    if not tpl_path.exists():
        raise SystemExit(f"error: template not found: {tpl_path}")

    out_prefix: Path
    if args.out_prefix is not None:
        out_prefix = Path(args.out_prefix)
        if not out_prefix.is_absolute():
            out_prefix = tomo_path.parent / out_prefix
    else:
        out_prefix = tomo_path.parent / tomo_path.stem

    print(f"loading tomogram: {tomo_path}")
    vol = load_mrc(tomo_path)
    print(f"  volume shape (z,y,x): {vol.shape}")
    print(f"loading template: {tpl_path}")
    template = load_mrc(tpl_path)
    print(f"  template shape (z,y,x): {template.shape}")

    bin_factor = max(1, int(args.bin))
    if bin_factor > 1:
        print(f"binning tomogram and template by factor {bin_factor}")
        vol_gpu = cp.asarray(vol, dtype=cp.float32)
        tpl_gpu = cp.asarray(template, dtype=cp.float32)
        vol_gpu = cskimage.transform.downscale_local_mean(vol_gpu, (bin_factor, bin_factor, bin_factor))
        tpl_gpu = cskimage.transform.downscale_local_mean(tpl_gpu, (bin_factor, bin_factor, bin_factor))
        vol = cp.asnumpy(vol_gpu).astype(np.float32)
        template = cp.asnumpy(tpl_gpu).astype(np.float32)
        voxel *= float(bin_factor)
        print(f"  binned volume shape (z,y,x): {vol.shape}")
        print(f"  binned template shape (z,y,x): {template.shape}")

        if args.debug:
            save_mrc(out_prefix.with_name(out_prefix.name + "_0_binned_tomogram.mrc"), vol, voxel_size=voxel)
            save_mrc(out_prefix.with_name(out_prefix.name + "_0_binned_template.mrc"), template, voxel_size=voxel)

    # 预处理：GPU 上做分位数归一化
    norm = normalize_percentile(vol, low=2.0, high=98.0)
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_1_norm.mrc"), norm, voxel_size=voxel)

    # gaussian smooth
    gauss = gaussian_smooth(norm, sigma=args.gauss_sigma)
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_2_gauss.mrc"), gauss, voxel_size=voxel)

    # vesselness
    print(f"computing vesselness: sigma in [{args.sigma_min}, {args.sigma_max}] with {args.sigma_steps} scales")
    vess = sato_vesselness(gauss, sigma_min=args.sigma_min, sigma_max=args.sigma_max, sigma_steps=args.sigma_steps)
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_3_sato.mrc"), vess, voxel_size=voxel)

    # skeleton 提取
    print("extracting skeleton curves from vesselness volume")
    curves_raw = extract_skeletons(
        vess,
        vesselness_percentile=float(args.vesselness_percentile),
        min_region_voxels=int(args.min_region_voxels),
        voxel_size=voxel,
        min_skel_length_A=float(args.min_skel_length),
        debug_prefix=out_prefix if args.debug else None,
    )
    print(f"  got {len(curves_raw)} raw skeleton curves")
    if not curves_raw:
        print("no skeleton found; nothing to do")
        return

    # 对每条骨架做一次平滑，作为“中心线曲线”
    curves: List[np.ndarray] = []
    for i, pts in enumerate(curves_raw):
        # 目标点数与近似长度相关：每 5 像素一个点
        if len(pts) >= 2:
            length_vox = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        else:
            length_vox = 1.0
        curve_points = max(20, int(length_vox / 5.0))
        smoothed = smooth_curve(pts, curve_points=curve_points, curvature=0.3)
        curves.append(smoothed.astype(np.float32))
        if args.debug:
            # 简单导出每条粗略曲线的 axm 以便快速浏览
            axm_path = out_prefix.parent / f"{out_prefix.name}_skel{i}.axm"
            write_axm(axm_path, smoothed, voxel_size=voxel)

    n_workers = max(1, min(int(args.workers), len(curves)))

    print(f"NCC matching on curves: workers={n_workers}")

    # 共享 tomogram
    shm = shared_memory.SharedMemory(create=True, size=vol.nbytes)
    results: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    try:
        shm_arr = np.ndarray(vol.shape, dtype=vol.dtype, buffer=shm.buf)
        shm_arr[:] = vol[:]
        del shm_arr

        cfg = {"block_step_px": float(args.block_step) / float(voxel)}
        tasks = list(enumerate(curves))

        if n_workers > 1:
            with multiprocessing.Pool(
                processes=n_workers,
                initializer=_init_match_worker,
                initargs=(shm.name, vol.shape, str(vol.dtype), template, cfg),
            ) as pool:
                for idx, curve_xyz, centers_xyz, scores in pool.imap_unordered(
                    _process_curve_worker, tasks
                ):
                    results[idx] = (curve_xyz, centers_xyz, scores)
        else:
            # 单进程（包括 GPU 模式）
            _init_match_worker(shm.name, vol.shape, str(vol.dtype), template, cfg)
            for idx, curve_xyz in tasks:
                ridx, c_xyz, centers_xyz, scores = _process_curve_worker((idx, curve_xyz))
                results[ridx] = (c_xyz, centers_xyz, scores)
            _worker_cleanup()
    finally:
        shm.close()
        shm.unlink()

    # 根据 NCC 得分筛选骨架
    accepted_indices: List[int] = []
    all_markers: List[Tuple[int, np.ndarray, float, bool]] = []
    for idx in range(len(curves)):
        curve_xyz, centers_xyz, scores = results.get(
            idx,
            (curves[idx], np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)),
        )
        if centers_xyz.size == 0:
            # 无有效匹配点，记入 debug 但不接受
            if args.debug:
                pass
            continue

        mean_score = float(np.mean(scores))
        good_mask = scores >= float(args.match_threshold)
        good_fraction = float(np.mean(good_mask.astype(np.float32)))
        accepted = (good_fraction >= float(args.good_fraction)) and (
            mean_score >= float(args.match_threshold) * 0.5
        )

        if args.debug:
            for c, s in zip(centers_xyz, scores):
                # 暂时不区分属于好/坏 fiber，最终用 accepted 标记
                all_markers.append((idx, c, float(s), accepted))

        if accepted:
            accepted_indices.append(idx)
            print(
                f"  curve {idx}: mean NCC={mean_score:.3f}, good_fraction={good_fraction:.3f} -> ACCEPT"
            )
        else:
            print(
                f"  curve {idx}: mean NCC={mean_score:.3f}, good_fraction={good_fraction:.3f} -> reject"
            )

    if not accepted_indices:
        print("no curves passed NCC thresholds; nothing to export")
        if args.debug and all_markers:
            cmm_path = out_prefix.with_name(out_prefix.name + "_debug.cmm")
            write_cmm_markers(cmm_path, all_markers)
            print(f"debug markers saved to {cmm_path}")
        return

    # 输出每条通过筛选的骨架曲线的 axm
    axm_paths: List[Path] = []
    for idx in accepted_indices:
        curve_xyz, _, _ = results[idx]
        axm_path = out_prefix.parent / f"{out_prefix.name}_fiber{idx}.axm"
        write_axm(axm_path, curve_xyz, voxel_size=voxel)
        axm_paths.append(axm_path)
    print(f"saved {len(axm_paths)} AXM files in {out_prefix.parent}")

    # 沿曲线按 spacing 采样粒子，并为每个点估计取向
    spacing_px = float(args.spacing) / float(voxel)
    all_coords: List[np.ndarray] = []
    all_angles: List[np.ndarray] = []
    for idx in accepted_indices:
        curve_xyz, _, _ = results[idx]
        pts = resample_curve_by_spacing(curve_xyz, spacing_px=spacing_px)
        if len(pts) == 0:
            continue
        # 切向量
        tangents = np.zeros_like(pts)
        if len(pts) == 1:
            tangents[0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            tangents[0] = pts[1] - pts[0]
            tangents[-1] = pts[-1] - pts[-2]
            if len(pts) > 2:
                tangents[1:-1] = 0.5 * (pts[2:] - pts[:-2])
        # 单位化并映射到欧拉角
        angs = np.zeros_like(pts)
        for i, t in enumerate(tangents):
            n = float(np.linalg.norm(t))
            if n < 1e-6:
                angs[i] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                rot, tilt, psi = vector_to_euler_zyz(t / n)
                angs[i] = np.array([rot, tilt, psi], dtype=np.float32)
        all_coords.append(pts.astype(np.float32))
        all_angles.append(angs.astype(np.float32))
        print(f"  curve {idx}: sampled {len(pts)} particles (spacing={args.spacing} Å)")

    if not all_coords:
        print("no particles sampled from accepted curves; STAR will not be written")
    else:
        coords = np.vstack(all_coords).astype(np.float32)
        angles = np.vstack(all_angles).astype(np.float32)
        star_path = out_prefix.with_name(out_prefix.name + "_particles.star")
        write_star(star_path, coords, angles)
        print(f"saved {len(coords)} particles in {star_path}")

    # debug CMM
    if args.debug and all_markers:
        cmm_path = out_prefix.with_name(out_prefix.name + "_debug.cmm")
        write_cmm_markers(cmm_path, all_markers)
        print(f"debug markers saved to {cmm_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

