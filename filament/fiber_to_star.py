#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纤维起止点 + .rec 体积 → 卷积/二值化/可调曲率拟合 → 颗粒 STAR 或 MOD 过滤转 STAR

本脚本实现了基于体积卷积的纤维自动拟合算法，主要功能：
1. 从 STAR 文件读取纤维起止点坐标和角度信息
2. 使用圆柱形核与 .rec 体积进行卷积，增强纤维信号
3. 通过二值化、侵蚀和骨架化提取纤维中心线
4. 使用可调曲率的样条曲线拟合纤维路径
5. 输出 .axm 格式的曲线（ArtiaX CurvedLine）用于可视化
6. 根据拟合曲线生成颗粒坐标（等间距采样或 MOD 过滤）

特点：
- 仅支持 .rec 体积输入（cryoET 断层图重建）
- --voxel-size 为必选参数，用于坐标尺度转换
- radius/spacing 单位为 Å（埃）
- 支持在 ChimeraX 中运行或独立命令行运行
- 参考算法：Automated picking of amyloid fibrils (FibrilFinder), PMC8217313
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import starfile
except ImportError:
    starfile = None
try:
    import mrcfile
except ImportError:
    mrcfile = None

# ========== ChimeraX 环境检测 ==========
def _in_chimerax():
    """
    检测脚本是否在 ChimeraX 环境中运行。
    
    ChimeraX 通过 runscript 命令运行脚本时，会在全局命名空间中注入一个 session 对象。
    该函数通过检查 globals() 中是否存在 session 对象来判断运行环境。
    
    返回:
        bool: 如果在 ChimeraX 中运行返回 True，否则返回 False
    """
    try:
        return "session" in globals() and globals()["session"] is not None
    except Exception:
        return False


def _get_session():
    """
    获取 ChimeraX 的 session 对象。
    
    session 对象提供了 ChimeraX 的核心功能接口，包括：
    - 命令执行（run commands）
    - 日志记录（logger）
    - 模型管理等
    
    返回:
        ChimeraX session 对象或 None（如果不在 ChimeraX 环境中）
    """
    return globals().get("session")


# ========== STAR 纤维起止点读取 ==========
def read_star_fibers(star_path: Path,) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], bool]:
    """
    读取纤维起止点 STAR 文件，提取纤维的起止坐标和角度信息。
    
    数据约定：
    - STAR 文件中每两行定义一条纤维：第一行为起始点，第二行为终止点
    - 必须包含列：_rlnCoordinateX, _rlnCoordinateY, _rlnCoordinateZ（或无下划线版本）
    - 可选包含列：_rlnAngleRot, _rlnAngleTilt, _rlnAnglePsi（欧拉角，ZYZ 惯例）
    - 若总行数为奇数，最后一行会被忽略（仅有起始点，无对应终止点）
    
    参数:
        star_path: STAR 文件路径
        
    返回:
        tuple: (fibers, has_orphan_start)
            - fibers: 列表，每个元素为 (start_xyz, end_xyz, start_angles, end_angles)
                - start_xyz: 起始点坐标 (x, y, z) ndarray，单位为像素
                - end_xyz: 终止点坐标 (x, y, z) ndarray，单位为像素
                - start_angles: 起始点欧拉角 (rot, tilt, psi) ndarray，单位为度
                - end_angles: 终止点欧拉角 (rot, tilt, psi) ndarray，单位为度
            - has_orphan_start: 若为 True，表示最后一行被忽略（行数为奇数）
    
    异常:
        ImportError: 若未安装 starfile 库
        ValueError: 若 STAR 文件缺少必需的坐标列
    """
    # 检查依赖库
    if starfile is None:
        raise ImportError("please install starfile: pip install starfile")
    
    # 读取 STAR 文件，always_dict=True 确保返回字典格式
    data = starfile.read(star_path, always_dict=True)
    # 提取数据表（可能包含多个 data block，取第一个）
    if isinstance(data, dict):
        df = next(iter(data.values()))
    else:
        df = data

    # 辅助函数：查找列名，支持 'rlnXXX' 或 '_rlnXXX' 或 'XXX' 格式
    def col(k):
        if k in df.columns:
            return k
        return None

    # 查找坐标列（必需）
    xc = col("_rlnCoordinateX") or col("rlnCoordinateX") or col("CoordinateX")
    yc = col("_rlnCoordinateY") or col("rlnCoordinateY") or col("CoordinateY")
    zc = col("_rlnCoordinateZ") or col("rlnCoordinateZ") or col("CoordinateZ")
    # 查找角度列（可选）
    ra = col("_rlnAngleRot") or col("rlnAngleRot") or col("AngleRot")
    ta = col("_rlnAngleTilt") or col("rlnAngleTilt") or col("AngleTilt")
    pa = col("_rlnAnglePsi") or col("rlnAnglePsi") or col("AnglePsi")

    # 验证必需列是否存在
    if not all([xc, yc, zc]):
        raise ValueError("input .star file is missing coordinate columns: CoordinateX/Y/Z")
    
    # 提取坐标数据（N 行 x 3 列）
    coords = df[[xc, yc, zc]].to_numpy(dtype=np.float32)

    # 提取角度数据（若不存在则填充为零）
    angles = np.zeros((len(df), 3), dtype=np.float32)
    if all([ra, ta, pa]):
        angles = df[[ra, ta, pa]].to_numpy(dtype=np.float32)

    # 将连续两行组合为一条纤维（起始点和终止点）
    fibers = []
    for i in range(0, len(coords) - 1, 2):
        fibers.append((coords[i], coords[i + 1], angles[i], angles[i + 1]))
    
    # 检查是否有孤立的起始点（行数为奇数）
    has_orphan = (len(coords) % 2) == 1
    return fibers, has_orphan


# ========== 体积加载与卷积拟合（参考 FibrilFinder 算法）==========
def load_volume(rec_path: Path) -> np.ndarray:
    """
    加载 .rec 体积文件（cryoET 断层图重建）。
    
    .rec 文件是 IMOD/SerialEM 输出的 MRC 格式体积数据，存储了 3D 断层图重建结果。
    该函数使用 mrcfile 库读取体积，并转换为 float32 精度以便后续数值计算。
    
    参数:
        rec_path: .rec 文件路径
        
    返回:
        np.ndarray: 3D 体积数组，形状为 (Z, Y, X)，数据类型为 float32
            - Z: 切片数量（深度方向）
            - Y: 每个切片的行数（高度方向）
            - X: 每个切片的列数（宽度方向）
    
    异常:
        ImportError: 若未安装 mrcfile 库
        
    注意:
        - permissive=True 允许读取不完全符合 MRC 标准的文件
        - mode="r" 以只读模式打开，避免意外修改原始数据
    """
    if mrcfile is None:
        raise ImportError("please install mrcfile: pip install mrcfile")
    with mrcfile.open(rec_path, permissive=True, mode="r") as m:
        return np.asarray(m.data, dtype=np.float32)


def save_mrc(path: Path, data: np.ndarray, voxel_size: float = 1.0):
    """保存数据为 MRC 文件。"""
    if mrcfile is None:
        raise ImportError("please install mrcfile: pip install mrcfile")
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(data.astype(np.float32))
        m.voxel_size = voxel_size


def crop_roi(vol: np.ndarray, start: np.ndarray, end: np.ndarray, margin_px: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从体积中裁剪包含纤维起止点的感兴趣区域（ROI）。
    
    该函数裁剪一个包围盒，使其包含从 start 到 end 的纤维段，并在四周留出 margin_px 的边距。
    裁剪 ROI 可以减少后续卷积和拟合的计算量，同时保留足够的上下文信息。
    
    坐标系统说明：
    - 输入坐标 start/end 为 (x, y, z)，这是 STAR/IMOD 的标准坐标系统
    - 体积数组 vol 的形状为 (Z, Y, X)，索引顺序为 [z, y, x]
    - 因此需要进行坐标顺序的转换
    
    参数:
        vol: 输入体积，形状为 (Z, Y, X)
        start: 纤维起始点坐标 (x, y, z)，单位为像素
        end: 纤维终止点坐标 (x, y, z)，单位为像素
        margin_px: 边距大小（像素），在起止点包围盒四周额外裁剪的范围
        
    返回:
        tuple: (roi, origin_xyz, shape)
            - roi: 裁剪的 3D 子体积，形状为 (Z', Y', X')
            - origin_xyz: ROI 在原始体积中的起始坐标 (x0, y0, z0)
            - shape: ROI 的形状数组 [nz, ny, nx]
            
    注意:
        - 裁剪区域会自动限制在体积边界内，避免越界
        - 返回的是原始数据的副本（.copy()），修改 ROI 不会影响原始体积
    """
    # 将输入转换为 numpy 数组
    s, e = np.asarray(start), np.asarray(end)
    
    # 计算包围盒的下界和上界（向下和向上取整以确保完全包含）
    lo = np.floor(np.minimum(s, e) - margin_px).astype(int)  # 最小坐标 - 边距
    hi = np.ceil(np.maximum(s, e) + margin_px).astype(int)   # 最大坐标 + 边距
    
    # 获取体积的实际尺寸（注意：vol.shape = (Z, Y, X)）
    nx, ny, nz = vol.shape[2], vol.shape[1], vol.shape[0]
    
    # 限制在体积边界内，避免越界访问
    lo = np.maximum(lo, [0, 0, 0])
    hi = np.minimum(hi, [nx - 1, ny - 1, nz - 1])
    
    # 提取裁剪范围（注意：Python 切片是左闭右开，所以 +1）
    x0, y0, z0 = lo[0], lo[1], lo[2]
    x1, y1, z1 = hi[0] + 1, hi[1] + 1, hi[2] + 1
    
    # 从体积中裁剪 ROI（坐标顺序转换：x,y,z → z,y,x）
    roi = vol[z0:z1, y0:y1, x0:x1].copy()
    
    # 记录 ROI 在全局坐标系中的起始位置
    origin = np.array([x0, y0, z0], dtype=np.float32)
    
    return roi, origin, np.array(roi.shape)


def rotation_matrix_from_z_to_vector(v: np.ndarray) -> np.ndarray:
    """
    计算将 z 轴 (0,0,1) 旋转到目标向量 v 的旋转矩阵。
    
    该函数用于纤维拟合中的坐标变换：将纤维方向对齐到 z 轴，便于使用沿 z 轴的圆柱核进行卷积。
    算法基于罗德里格斯（Rodrigues）旋转公式，通过叉积和点积计算旋转轴和旋转角。
    
    数学原理：
    - 旋转轴 a = u × v（u 为初始向量 [0,0,1]，v 为目标向量）
    - 旋转角通过 cos(θ) = u·v 确定
    - 罗德里格斯公式：R = I + [a]× + [a]×² * (1/(1+cos(θ)))
      其中 [a]× 是旋转轴 a 的反对称矩阵（skew-symmetric matrix）
    
    特殊情况处理：
    1. 若 v ≈ -z（反向），叉积接近零，直接返回 180° 旋转矩阵 diag(-1,-1,1)
    2. 若 v ≈ z（同向），叉积接近零，直接返回单位矩阵（无需旋转）
    
    参数:
        v: 目标方向向量 (3,)，可以是任意长度（函数内会归一化）
        
    返回:
        np.ndarray: 3x3 旋转矩阵 R，使得 R @ [0,0,1] ≈ v/||v||
    """
    # 归一化目标向量（防止除零）
    v = np.asarray(v, dtype=np.float32).ravel()[:3]
    v = v / (np.linalg.norm(v) + 1e-12)
    
    # 特殊情况 1：v ≈ -z（向量几乎反向）
    # 此时叉积会接近 0，无法确定唯一旋转轴，单独处理
    if np.abs(v[2] + 1) < 1e-6:
        # 绕任意垂直轴旋转 180°，这里选择绕 x 或 y 轴
        return np.diag([-1.0, -1.0, 1.0])
    
    # 初始向量（z 轴单位向量）
    u = np.array([0, 0, 1], dtype=np.float32)
    
    # 计算旋转轴：a = u × v（叉积）
    a = np.cross(u, v)
    
    # 计算旋转角的余弦：c = u·v（点积）
    c = np.dot(u, v)
    
    # 特殊情况 2：v ≈ z（向量几乎同向）
    # 此时 a ≈ 0，无需旋转
    if np.linalg.norm(a) < 1e-12:
        return np.eye(3)
    
    # 构造反对称矩阵（skew-symmetric matrix）[a]×
    # [a]× = [[0, -az, ay],
    #         [az, 0, -ax],
    #         [-ay, ax, 0]]
    ax = np.array([[0, -a[2], a[1]], 
                   [a[2], 0, -a[0]], 
                   [-a[1], a[0], 0]])
    
    # 罗德里格斯公式：R = I + [a]× + [a]×² * (1/(1+cos(θ)))
    R = np.eye(3) + ax + ax @ ax * (1.0 / (1.0 + c))
    
    return R


def apply_rotation_to_volume(vol: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """
    对 3D 体积应用旋转变换（绕体积中心旋转）。
    该函数用于将纤维 ROI 旋转到标准方向（纤维轴沿 z 轴），便于使用标准的圆柱核进行卷积。
    旋转中心固定在体积的几何中心。
    坐标系统转换：
    - 输入旋转矩阵 R 作用于 (x, y, z) 坐标系统
    - 体积数组存储顺序为 (Z, Y, X)，索引顺序为 [z, y, x]
    - scipy.ndimage.affine_transform 使用 (z, y, x) 索引顺序
    - 因此需要对 R 进行坐标轴重排：x,y,z → z,y,x
    
    参数:
        vol: 输入体积，形状为 (Z, Y, X)
        R: 3x3 旋转矩阵，作用于 (x, y, z) 坐标系统
        origin: vol 在全局坐标系中的起始位置 (x0, y0, z0)（当前实现中未使用）
        
    返回:
        np.ndarray: 旋转后的体积，形状与输入相同
        
    注意:
        - 使用双线性插值（order=1）平衡速度和精度
        - 旋转中心在体积几何中心，与 origin 无关
    """
    from scipy.ndimage import affine_transform
    
    shape = vol.shape  # (Z, Y, X)
    
    # 计算体积中心点（在 z,y,x 坐标下）
    # 对于长度为 n 的维度，中心索引为 (n-1)/2
    center = np.array([(shape[0] - 1) / 2.0,  # z 中心
                       (shape[1] - 1) / 2.0,  # y 中心
                       (shape[2] - 1) / 2.0]) # x 中心
    
    # 构造 4x4 齐次变换矩阵（虽然只使用 3x3 旋转部分）
    R_in = np.eye(4)
    R_in[:3, :3] = R.T  # 转置是因为 affine_transform 使用逆变换
    
    # 计算偏移量：先平移到原点，旋转，再平移回中心
    # offset = center - R^T @ center
    offset = center - np.dot(R.T, center)
    
    # 将旋转矩阵从 (x,y,z) 顺序转换为 (z,y,x) 顺序
    # scipy.ndimage.affine_transform 期望的矩阵作用于索引 (z,y,x)
    R_zyx = np.eye(3)
    R_zyx[0, :] = R_in[2, :3]  # z 分量 <- R 的 x 行
    R_zyx[1, :] = R_in[1, :3]  # y 分量 <- R 的 y 行
    R_zyx[2, :] = R_in[0, :3]  # x 分量 <- R 的 z 行
    
    # 应用仿射变换（旋转 + 平移）
    # order=1: 双线性插值（平衡速度和精度）
    rotated = affine_transform(vol, R_zyx, offset=offset, order=1, mode="reflect")
    
    return rotated


# ========== 卷积前预处理（归一化 + Gaussian + Kuwahara + Bottom-hat）==========

def _normalize_roi(roi: np.ndarray, percentile_low: float = 2.0, percentile_high: float = 98.0, use_std: bool = True) -> np.ndarray:
    """
    对 ROI 体积做稳健归一化，使不同纤维/ROI 的尺度一致，便于后续滤波与卷积。
    percentile_low, percentile_high: 归一化前先做百分位裁剪，避免极值拉偏均值和标准差。
        - 范围 (0, 100)。(2, 98) 较稳健；(0, 100) 不裁剪。
    use_std: True 时做 (v - mean) / std；False 时仅减均值。
        - 若 std 接近 0（平坦区域），会退化为仅减均值并避免除零。
    """
    out = np.asarray(roi, dtype=np.float32)
    valid = np.isfinite(out)
    if not np.any(valid):
        return out

    v = out[valid]
    lo = np.percentile(v, percentile_low)
    hi = np.percentile(v, percentile_high)
    out = np.clip(out, lo, hi)
    mean = np.mean(out[valid])
    if use_std:
        std = np.std(out[valid])
        if std > 1e-12:
            out = (out - mean) / std
        else:
            out = out - mean
    else:
        out = out - mean
    return out.astype(np.float32)


def _gaussian_filter_3d(roi: np.ndarray, sigma: float) -> np.ndarray:
    """
    3D 高斯滤波，提高信噪比、平滑噪声。
        sigma: 高斯标准差（像素）。越大越平滑、纤维边缘越糊；过小则降噪不足。
            - 可与纤维半径成比例，例如 0.5 * radius_px 或固定 1~2，原文320px的micrograph使用 24 px。
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(roi, sigma=sigma, mode="reflect").astype(np.float32)


def _kuwahara_2d(slice_2d: np.ndarray, window_radius: int) -> np.ndarray:
    """
    2D Kuwahara：将每个像素替换为 4 个重叠象限中方差最小的那个的均值。
    用于按 z 切片调用以替代 3D generic_filter（3D 逐体素 Python 回调极慢）。
    """
    from scipy.ndimage import generic_filter
    n = 2 * window_radius + 1

    def _quadrant_fn(values: np.ndarray) -> float:
        v = values.reshape(n, n)
        r = window_radius
        best_mean = float(values[0])
        best_var = np.inf
        # 4 个重叠象限（标准 Kuwahara 划分）
        for iy in (0, 1):
            for ix in (0, 1):
                y0, y1 = (0, r + 1) if iy == 0 else (r + 1, n)
                x0, x1 = (0, r + 1) if ix == 0 else (r + 1, n)
                region = v[y0:y1, x0:x1]
                if region.size == 0:
                    continue
                m = float(region.mean())
                var = float(region.var())
                if var < best_var:
                    best_var = var
                    best_mean = m
        return best_mean

    return generic_filter(slice_2d.astype(np.float64), _quadrant_fn, size=(n, n), mode="reflect").astype(np.float32)


def _kuwahara_3d(roi: np.ndarray, window_radius: int) -> np.ndarray:
    """
    3D 体积上的 Kuwahara：按 z 切片做 2D Kuwahara 再堆叠。
    纯 3D 八分体 + generic_filter 会逐体素调用 Python，百万次量级导致数分钟无输出，
    可调参数：window_radius，同 _kuwahara_2d，原文320px的micrograph使用 60 px。
    """
    out = np.empty_like(roi, dtype=np.float32)
    for z in range(roi.shape[0]):
        out[z] = _kuwahara_2d(roi[z], window_radius)
    return out


def _bottom_hat_3d(roi: np.ndarray, structure_radius: int) -> np.ndarray:
    """
    3D 底帽变换：closing(roi) - roi，用于去除缓慢强度变化、突出比背景暗的结构（纤维）。
        structure_radius: 形态学结构元半径（像素）。越大去除的“缓慢变化”尺度越大，
            - 纤维与背景的对比越干净；过大会模糊纤维边界。
            - 3D 中可用较小值，例如 3~8，原文320px的micrograph使用 300 px。
    """
    from scipy.ndimage import grey_closing
    try:
        from skimage.morphology import ball
        structure = ball(structure_radius)
    except ImportError:
        # 无 skimage 时用立方体结构元
        print("warning: skimage not found, using cubic structure element.")
        s = 2 * structure_radius + 1
        structure = np.ones((s, s, s), dtype=np.float64)
    closed = grey_closing(roi.astype(np.float64), structure=structure, mode="reflect")
    # 底帽 = closing - 原图；纤维（暗）处 closing > 原图，底帽为正，纤维在预处理后为“亮”便于卷积
    out = closed - roi.astype(np.float64)
    return out.astype(np.float32)


def make_cylinder_kernel(radius_px: float, length_px: float) -> np.ndarray:
    """
    生成 3D 圆柱形卷积核（轴沿 z 方向）。
    
    该卷积核用于与体积卷积，增强纤维状结构的信号。圆柱形状匹配纤维的几何特征，
    当圆柱核的轴与纤维对齐时，卷积响应最强。这是 FibrilFinder 算法的核心技术。
    
    圆柱定义：
    - 轴方向：z 轴
    - 半径：在 x-y 平面内，到 z 轴的距离 ≤ radius_px
    - 长度：z 方向上，|z| ≤ length_px/2
    
    归一化：
    - 核内所有体素值之和为 1（或接近 1），使卷积结果的量纲一致
    
    参数:
        radius_px: 圆柱半径（像素），对应纤维的粗细
        length_px: 圆柱长度（像素），影响卷积的局部性
        
    返回:
        np.ndarray: 3D 归一化的圆柱核，形状约为 (2*half+1, 2*r+3, 2*r+3)
            - 核内（圆柱内）体素值为 1/N（N 为圆柱体素总数）
            - 核外体素值为 0
            
    注意:
        - 半径和长度会向上取整以确保至少包含几个体素
        - x-y 方向的范围留出 1 个体素的余量（r+2）以覆盖边界
    """
    # 确保半径至少为 1 像素（向上取整）
    r = max(1, int(round(radius_px)))
    
    # 长度的一半（圆柱从 -half 到 +half）
    half = max(1, int(round(length_px / 2)))
    
    # 创建 z, y, x 方向的坐标数组
    zz = np.arange(-half, half + 1, dtype=np.float32)  # z: [-half, half]
    yy = np.arange(-r - 1, r + 2, dtype=np.float32)    # y: [-r-1, r+1]，留余量
    xx = np.arange(-r - 1, r + 2, dtype=np.float32)    # x: [-r-1, r+1]，留余量
    
    # 生成 3D 网格（indexing="ij" 确保顺序为 Z, Y, X）
    Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
    
    # 计算每个点到 z 轴的距离（在 x-y 平面内的径向距离）
    R = np.sqrt(X**2 + Y**2)
    
    # 圆柱掩码：径向距离 ≤ radius_px 且 z 方向距离 ≤ length_px/2
    kernel = ((R <= radius_px) & (np.abs(Z) <= length_px / 2)).astype(np.float32)
    
    # 归一化：使核内所有体素之和为 1
    # max(..., 1) 防止空核导致除零错误
    kernel /= max(kernel.sum(), 1)
    
    return kernel


def fit_fiber_convolve(
    vol: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    radius_px: float,
    threshold_percentile: float = 75.0,
    curvature: float = 0.1,
    erode_iterations: int = 0,
    curve_points: int = 50,
    debug_mrc_prefix: Optional[str] = None,
    voxel_size: float = 1.0,
    length_scale: float = 0.5,
    norm_percentile_low: float = 2.0,
    norm_percentile_high: float = 98.0,
    gaussian_sigma: Optional[float] = None,
    kuwahara_radius: int = 3,
    bottomhat_radius: int = 5,
    convolution_mode: str = "reflect",
) -> np.ndarray:
    """
    使用卷积方法拟合纤维的 3D 路径（基于 FibrilFinder 算法）。
    
    算法流程：
    1. ROI 裁剪：提取包含纤维的局部区域
    2. 坐标旋转：将纤维方向对齐到 z 轴
    3. 预处理：归一化 → 3D Gaussian → 3D Kuwahara → 3D Bottom-hat（纤维为暗结构）
    4. 卷积增强：用圆柱核卷积，增强纤维状结构
    5. 二值化 → 侵蚀（可选）→ 骨架化 → 样条拟合
    
    参数:
        vol, start, end, radius_px: 体积与纤维起止点、半径（像素）
        threshold_percentile: 二值化百分位 (0-100)，越高越严格
        curvature: 曲率 [0,1]，0 最直、1 最弯
        length_scale: ROI 边距 = radius_px * length_scale
        erode_iterations: 形态学侵蚀次数，>0 可去细小分支
        norm_percentile_low, norm_percentile_high: 归一化前百分位裁剪，避免极值拉偏。
            - (2, 98) 稳健；(0, 100) 不裁剪。
        gaussian_sigma: 高斯滤波标准差（像素）。None 时用 max(1.0, 0.5*radius_px)。
            - 越大越平滑、边缘越糊；过小降噪不足。
        kuwahara_radius: 3D Kuwahara 窗口半宽（像素），窗口 (2*radius+1)^3。
            - 越大越平滑、边缘保留越好但更慢；建议与纤维半径同量级。
        bottomhat_radius: 3D 底帽结构元半径（像素）。越大去除缓慢变化越强。
            - 过大会模糊纤维边界。
        convolution_mode: 卷积边界模式。"reflect" 减轻棱边伪影；可选 "nearest", "constant" 等。
    """
    from scipy.ndimage import convolve
    from scipy.interpolate import UnivariateSpline

    # ========== 步骤 1: ROI 裁剪 ==========
    margin = radius_px * length_scale
    roi, origin, shp = crop_roi(vol, start, end, margin)
    if roi.size == 0:
        print("warning: ROI is empty, returning straight line.")
        return np.array([start, end])

    # ========== 步骤 2: 坐标旋转 ==========
    v = end - start
    v = v / (np.linalg.norm(v) + 1e-12)
    R = rotation_matrix_from_z_to_vector(v)
    roi_rot = apply_rotation_to_volume(roi, R, origin)

    # ========== 步骤 2.5: 卷积前预处理（归一化 + Gaussian + Kuwahara + Bottom-hat）==========
    # 归一化：使不同 ROI 尺度一致，减轻“只有棱边有信号”
    roi_pre = _normalize_roi(roi_rot, percentile_low=norm_percentile_low, percentile_high=norm_percentile_high, use_std=True,)
    if debug_mrc_prefix is not None:  # 1
        save_mrc(f"{debug_mrc_prefix}_norm.mrc", roi_pre, voxel_size)

    # 3D Gaussian：提高信噪比。可调：gaussian_sigma 越大越平滑
    sigma = gaussian_sigma if gaussian_sigma is not None else max(1.0, 0.5 * radius_px)
    roi_pre = _gaussian_filter_3d(roi_pre, sigma)
    if debug_mrc_prefix is not None:  # 2
        save_mrc(f"{debug_mrc_prefix}_gaussian.mrc", roi_pre, voxel_size)

    # 3D Kuwahara：降噪且保留边缘。可调：kuwahara_radius 越大越平滑、计算越慢
    roi_pre = _kuwahara_3d(roi_pre, kuwahara_radius)
    if debug_mrc_prefix is not None:  # 3
        save_mrc(f"{debug_mrc_prefix}_kuwahara.mrc", roi_pre, voxel_size)
        
    # 3D Bottom-hat：突出比背景暗的纤维。可调：bottomhat_radius 越大去缓慢变化越强
    roi_pre = _bottom_hat_3d(roi_pre, bottomhat_radius)
    if debug_mrc_prefix is not None:  # 4
        save_mrc(f"{debug_mrc_prefix}_bottomhat.mrc", roi_pre, voxel_size)

    # ========== 步骤 3: 卷积增强 ==========
    length_px = max(4, radius_px * 2)
    kernel = make_cylinder_kernel(radius_px, length_px)
    # 使用 reflect 减轻 ROI 棱边虚假高响应；可调：convolution_mode 可选 "nearest", "constant" 等
    try:
        if convolution_mode == "constant":
            resp = convolve(roi_pre, kernel, mode="constant", cval=0.0)
        else:
            resp = convolve(roi_pre, kernel, mode=convolution_mode)
    except Exception:
        print("warning: convolution failed, using preprocessed volume.")
        resp = roi_pre

    if debug_mrc_prefix is not None:  # 5
        save_mrc(f"{debug_mrc_prefix}_conv.mrc", resp, voxel_size)

    # ========== 步骤 4: 二值化 ==========
    # 根据百分位阈值将卷积响应转为二值掩码
    # threshold_percentile: 高于此百分位的体素被视为纤维（0-100，默认 75）
    # 值越高越严格，只保留最强的信号
    th = np.percentile(resp[np.isfinite(resp)], threshold_percentile)
    binary = (resp >= th).astype(np.uint8)

    if debug_mrc_prefix is not None:  # 6
        save_mrc(f"{debug_mrc_prefix}_binary.mrc", binary, voxel_size)

    # ========== 步骤 5: 侵蚀（可选）==========
    # 形态学侵蚀：去除细小分支和噪声，使纤维更连续
    # erode_iterations > 0 时执行，纤维信号不连续时可尝试 1-3 次迭代
    if erode_iterations > 0:
        try:
            from scipy.ndimage import binary_erosion
            # 3x3x3 立方体结构元素（26-连通）
            struct = np.ones((3, 3, 3), dtype=np.uint8)
            for _ in range(erode_iterations):
                binary = binary_erosion(binary, structure=struct).astype(np.uint8)
        except Exception:
            # 侵蚀失败时跳过（不影响后续流程）
            print("warning: erosion failed, skipping.")
            pass
    
    if debug_mrc_prefix is not None:  # 7
        save_mrc(f"{debug_mrc_prefix}_binary_eroded.mrc", binary, voxel_size)

    # ========== 步骤 6: 骨架化 ==========
    # 提取二值掩码的中心线（3D skeleton），骨架点即为纤维中心线
    try:
        from skimage.morphology import skeletonize
        # 3D 骨架化：将二值体积细化为单像素宽的中心线（skeletonize 支持 2D/3D，替代已弃用的 skeletonize_3d）
        skel = skeletonize(binary)
        # 提取骨架点坐标 (z,y,x)
        pts_zyx = np.argwhere(skel > 0)
    except ImportError:
        # 若无 scikit-image，退化为简单峰值检测（精度较低）
        # 对每个 (y,x) 列，取卷积响应最大的 z 位置作为中心线近似
        # 建议安装 scikit-image 以获得真正的 3D 骨架中心线
        print("warning: skeletonization failed, using simple peak detection.")
        pts_zyx = []
        for iy in range(resp.shape[1]):
            for ix in range(resp.shape[2]):
                col = resp[:, iy, ix]  # 提取 z 方向的列
                if np.any(binary[:, iy, ix]):  # 如果该列有纤维信号
                    iz = np.argmax(col)  # 找到响应最大的 z 位置
                    pts_zyx.append([iz, iy, ix])
        # 若未找到任何点，使用体积中心作为退化情况
        pts_zyx = np.array(pts_zyx, dtype=np.float32) if pts_zyx else np.array([[resp.shape[0] // 2, resp.shape[1] // 2, resp.shape[2] // 2]])

    # 若骨架化后未找到任何点，使用体积中心
    if len(pts_zyx) == 0:
        print("warning: no skeleton points found, using volume center.")
        pts_zyx = np.array([[resp.shape[0] // 2, resp.shape[1] // 2, resp.shape[2] // 2]])
    
    if debug_mrc_prefix is not None:  # 8
        save_mrc(f"{debug_mrc_prefix}_pts_zyx.mrc", pts_zyx, voxel_size)

    # 将骨架点从旋转后的 ROI 坐标转换回全局坐标
    # 步骤：(z,y,x) → (x,y,z) → 去中心化 → 逆旋转 → 平移到全局
    # 必须用 ROI 几何中心在全局的坐标（origin + center_rot），不能用纤维中点 mid，
    # 否则骨架点会整体偏移；裁剪后曲线首尾会在步骤 7 强制设为 start/end。
    pts_xyz_rot = np.column_stack([pts_zyx[:, 2], pts_zyx[:, 1], pts_zyx[:, 0]])
    center_rot = np.array([(shp[2] - 1) / 2.0, (shp[1] - 1) / 2.0, (shp[0] - 1) / 2.0])
    pts_xyz_rot_centered = pts_xyz_rot - center_rot
    roi_center_global = origin + center_rot  # ROI 几何中心在全局坐标
    pts_ordered = (R.T @ pts_xyz_rot_centered.T).T + roi_center_global
    # 5. 按到起点 start 的距离重排骨架点（与纤维 index 无关，仅使中心线沿起点→终点有序）
    dist_to_start = np.linalg.norm(pts_ordered - start, axis=1)
    order = np.argsort(dist_to_start)
    pts_ordered = pts_ordered[order]

    # 去重与子采样：移除过于接近的点，使样条拟合更稳定
    # 保持原有顺序，只做邻近去重（距离 < 1 像素的点会被跳过）
    if len(pts_ordered) > 2:
        keep = [0]  # 保留第一个点
        for i in range(1, len(pts_ordered)):
            # 如果当前点与上一个保留点距离 >= 1 像素，则保留
            if np.linalg.norm(pts_ordered[i] - pts_ordered[keep[-1]]) >= 1.0:
                keep.append(i)
        # 确保最后一个点也被保留
        if keep[-1] != len(pts_ordered) - 1:
            keep.append(len(pts_ordered) - 1)
        pts_ordered = pts_ordered[keep]
    
    # 若去重后点数不足，无法拟合，返回直线段
    if len(pts_ordered) < 2:
        print("warning: not enough points after deduplication, returning straight line.")
        return np.array([start, end])

    # ========== 步骤 7: 样条拟合 ==========
    # 用可调曲率的样条曲线平滑拟合骨架点
    # curvature ∈ [0,1]: 0=最直（强平滑），1=最弯（紧贴骨架点）
    # 计算骨架点的累积弧长参数（用于样条参数化）
    arc = np.zeros(len(pts_ordered))
    arc[1:] = np.cumsum(np.linalg.norm(np.diff(pts_ordered, axis=0), axis=1))
    
    # 若总弧长太短（接近 0），无法拟合，返回直线段
    if arc[-1] < 1e-6:
        print("warning: too short arc length, returning straight line.")
        return np.array([start, end])
    
    # 输出曲线的采样点数（由 curve_points 控制，至少 5 点以保证平滑）
    n_pts = max(5, min(curve_points, len(pts_ordered) * 2))
    
    # 计算平滑参数 s（UnivariateSpline 的 s 越大越平滑）
    # s = (1 - curvature) * arc_length^2
    # - curvature=0 → s 最大 → 最平滑（直线）
    # - curvature=1 → s=0 → 无平滑（紧贴数据点）
    curv = np.clip(curvature, 0.0, 1.0)
    s_smooth = (1.0 - curv) * (arc[-1] ** 2)
    
    # 对 x, y, z 分别拟合一维样条曲线
    # k: 样条次数，最大为 3（三次样条），若点数不足则降阶
    try:
        sp_x = UnivariateSpline(arc, pts_ordered[:, 0], s=s_smooth, k=min(3, len(pts_ordered) - 1))
        sp_y = UnivariateSpline(arc, pts_ordered[:, 1], s=s_smooth, k=min(3, len(pts_ordered) - 1))
        sp_z = UnivariateSpline(arc, pts_ordered[:, 2], s=s_smooth, k=min(3, len(pts_ordered) - 1))
    except Exception:
        # 样条拟合失败时返回直线段
        print("warning: spline fitting failed, returning straight line.")
        return np.array([start, end])
    
    # 在弧长参数上均匀采样，生成最终曲线
    u = np.linspace(0, arc[-1], n_pts)
    curve = np.column_stack([sp_x(u), sp_y(u), sp_z(u)])
    # 强制首尾为给定的纤维起止点（变换后骨架在 ROI 内，但样条可能略偏离；此处保证曲线必过 start/end）
    curve[0], curve[-1] = start, end
    return curve


def resample_curve_by_spacing(curve: np.ndarray, spacing_px: float) -> np.ndarray:
    """
    沿曲线按固定间距重采样点（用于等间距颗粒采样）。
    
    该函数计算曲线的弧长参数，然后按指定间距在弧长上均匀采样，
    通过线性插值获得新的点坐标。这确保颗粒沿纤维均匀分布。
    
    参数:
        curve: 输入曲线，形状为 (N, 3)，每行为一个点 (x, y, z)
        spacing_px: 采样间距（像素），两个相邻颗粒之间的距离
        
    返回:
        np.ndarray: 重采样后的曲线，形状为 (M, 3)
            - M = floor(total_length / spacing_px)
            - 若曲线长度为 0，返回第一个点
            
    注意:
        - 使用线性插值（kind="linear"），速度快且精度足够
        - fill_value="extrapolate" 允许轻微外推（通常不会触发）
    """
    from scipy.interpolate import interp1d
    
    # 计算累积弧长参数
    d = np.zeros(len(curve))
    d[1:] = np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
    total = d[-1]  # 总弧长
    
    # 若曲线长度为 0，返回第一个点
    if total <= 0:
        return curve[:1]
    
    # 计算采样点数：总长度 / 间距
    n_pts = max(1, int(round(total / spacing_px)))
    
    # 在弧长参数上均匀采样
    s = np.linspace(0, total, n_pts, endpoint=True)
    
    # 对 x, y, z 分别进行线性插值
    fx = interp1d(d, curve[:, 0], kind="linear", fill_value="extrapolate")
    fy = interp1d(d, curve[:, 1], kind="linear", fill_value="extrapolate")
    fz = interp1d(d, curve[:, 2], kind="linear", fill_value="extrapolate")
    
    # 返回重采样的点
    return np.column_stack([fx(s), fy(s), fz(s)])


def distance_point_to_curve(point: np.ndarray, curve: np.ndarray) -> float:
    """
    计算点到折线（polyline）的最短距离（用于 MOD 过滤）。
    
    该函数计算点到曲线上每个线段的最短距离，然后返回所有距离中的最小值。
    对于每个线段，考虑三种情况：
    1. 点在线段端点处最近（投影在端点外）
    2. 点在线段中间某处最近（投影在线段内）
    
    算法：
    - 首先计算点到所有顶点的距离
    - 然后对每个线段，计算点到该线段的投影
    - 若投影在线段内（0 ≤ t ≤ 1），计算点到投影点的距离
    - 返回所有距离中的最小值
    
    参数:
        point: 查询点坐标 (x, y, z)
        curve: 折线顶点，形状为 (N, 3)，每行为一个顶点 (x, y, z)
        
    返回:
        float: 点到折线的最短距离（像素）
        
    复杂度:
        O(N)，其中 N 为曲线顶点数
    """
    # 计算点到所有顶点的距离
    d = np.linalg.norm(curve - point, axis=1)
    
    # 计算所有线段（相邻顶点之间的向量）
    segs = np.diff(curve, axis=0)  # (N-1, 3)
    
    # 计算每个线段的长度
    lens = np.linalg.norm(segs, axis=1)
    lens[lens < 1e-10] = 1e-10  # 防止除零（对于重复顶点）
    
    # 对每个线段，计算点到该线段的最短距离
    for i in range(len(segs)):
        # 计算点在线段上的投影参数 t ∈ [0, 1]
        # t = (point - seg_start) · seg_direction / ||seg||^2
        t = np.dot(point - curve[i], segs[i]) / (lens[i] ** 2)
        t = np.clip(t, 0, 1)  # 限制在线段范围内
        
        # 计算投影点坐标
        proj = curve[i] + t * segs[i]
        
        # 更新最短距离
        d = np.minimum(d, np.linalg.norm(point - proj))
    
    return float(np.min(d))


def filter_points_near_curve(points: np.ndarray, curve: np.ndarray, radius_px: float) -> np.ndarray:
    """
    过滤 MOD 点：仅保留到曲线距离 ≤ radius_px 的点（MOD 过滤模式）。
    
    该函数用于从手动挑选的颗粒（MOD 文件）中筛选出位于拟合纤维曲线附近的颗粒。
    这样可以去除误挑选的点，只保留真正属于该纤维的颗粒。
    
    参数:
        points: MOD 中的所有点，形状为 (N, 3)
        curve: 拟合的纤维曲线，形状为 (M, 3)
        radius_px: 距离阈值（像素），通常设为纤维半径
            - 点到曲线的距离 ≤ radius_px 时保留
            - 点到曲线的距离 > radius_px 时过滤
        
    返回:
        np.ndarray: 过滤后的点，形状为 (K, 3)，K ≤ N
            - 若无点满足条件，返回空数组 (0, 3)
            
    复杂度:
        O(N * M)，其中 N 为点数，M 为曲线顶点数
        
    注意:
        - 该操作对每个点都调用 distance_point_to_curve，可能较慢
        - 若 MOD 中点数很多，建议增大 radius_px 或减少曲线顶点数
    """
    # 对每个点计算到曲线的距离，保留距离 ≤ radius_px 的点
    keep = [i for i in range(len(points)) if distance_point_to_curve(points[i], curve) <= radius_px]
    # 返回过滤后的点（若无点满足条件，返回空数组）
    return points[np.array(keep)] if keep else np.zeros((0, 3), dtype=np.float32)


# ========== MOD 文件读取（IMOD 格式）==========
def read_mod_points(mod_path: Path) -> np.ndarray:
    """
    读取 IMOD .mod 文件中所有轮廓点（颗粒坐标）。
    
    IMOD .mod 文件是 IMOD 软件（3dmod）生成的模型文件，用于存储手动挑选的颗粒坐标。
    文件格式为二进制，结构如下：
    - 文件头：'IMOD' 标识 + 模型元数据
    - 对象（Object）：每个对象包含多个轮廓（Contour）
    - 轮廓（Contour）：每个轮廓包含多个点（Point）
    - 点坐标：float32 格式的 (x, y, z)
    
    该函数解析所有对象和轮廓，提取其中的所有点坐标。
    
    参数:
        mod_path: .mod 文件路径
        
    返回:
        np.ndarray: 所有点坐标，形状为 (N, 3)，每行为 (x, y, z)
            - 若文件中无点，返回空数组 (0, 3)
            - 坐标单位为像素
            
    异常:
        ValueError: 若不是有效的 IMOD 文件（缺少 'IMOD' 标识）
        
    注意:
        - 使用大端字节序（big-endian, '>i', '>f4'）
        - 仅提取点坐标，忽略其他元数据（如颜色、标签等）
        
    参考:
        - IMOD file format: https://bio3d.colorado.edu/imod/doc/binspec.html
    """
    mod_path = Path(mod_path)
    with open(mod_path, "rb") as f:
        # 验证文件标识（前 4 字节必须为 'IMOD'）
        if f.read(4) != b"IMOD":
            raise ValueError(f"not a valid IMOD file: {mod_path}")
        
        # 跳过版本号（4 字节）
        f.read(4)
        
        # 读取文件头（232 字节）
        header = f.read(232)
        # 从头部提取对象数量（偏移 140 字节，4 字节整数，大端）
        objsize = struct.unpack(">i", header[140:144])[0]
        
        # 存储所有点的列表
        all_pts: List[np.ndarray] = []
        
        # 遍历所有对象
        for _ in range(objsize):
            # 检查对象标识（'OBJT'）
            if f.read(4) != b"OBJT":
                break
            
            # 读取对象数据（176 字节）
            obj_data = f.read(176)
            # 从对象数据中提取轮廓数量（偏移 128 字节，4 字节整数）
            contsize = struct.unpack(">i", obj_data[128:132])[0]
            
            # 遍历该对象的所有轮廓
            for _ in range(contsize):
                # 检查轮廓标识（'CONT'）
                if f.read(4) != b"CONT":
                    break
                
                # 读取轮廓中的点数（4 字节整数）
                psize = struct.unpack(">i", f.read(4))[0]
                
                # 跳过轮廓元数据（12 字节）
                f.read(12)
                
                # 读取点坐标数据（每个点 12 字节 = 3 * float32）
                buf = f.read(12 * psize)
                if len(buf) != 12 * psize:
                    break
                
                # 解析坐标：大端 float32 → (psize, 3) → float32
                pts = np.frombuffer(buf, dtype=">f4").reshape(psize, 3).astype(np.float32)
                all_pts.append(pts)
            
            # 跳过对象的其他可选数据块（如果有）
            while True:
                peek = f.read(4)
                # 检查是否到达下一个对象或文件结束
                if len(peek) < 4 or peek == b"OBJT" or peek == b"IEOF":
                    if len(peek) == 4 and peek == b"OBJT":
                        # 回退到对象标识前，以便下一轮循环读取
                        f.seek(-4, 1)
                    break
                # 读取数据块大小并跳过
                sz = f.read(4)
                if len(sz) < 4:
                    break
                f.seek(struct.unpack(">i", sz)[0], 1)
        
        # 若未找到任何点，返回空数组
        if not all_pts:
            return np.zeros((0, 3), dtype=np.float32)
        
        # 合并所有点为单个数组
        return np.vstack(all_pts)


# ========== 输出函数（STAR 和 ArtiaX .axm 格式）==========
def write_star(out_path: Path, coords: np.ndarray, angles: np.ndarray, origins: Optional[np.ndarray] = None):
    # 若未提供原点偏移，填充为全 0
    if origins is None:
        origins = np.zeros((len(coords), 3), dtype=np.float32)
    
    import pandas as pd
    
    # 构造 DataFrame（RELION 标准列名）
    df = pd.DataFrame({
        "rlnCoordinateX": coords[:, 0], 
        "rlnCoordinateY": coords[:, 1], 
        "rlnCoordinateZ": coords[:, 2],
        "rlnOriginX": origins[:, 0], 
        "rlnOriginY": origins[:, 1], 
        "rlnOriginZ": origins[:, 2],
        "rlnAngleRot": angles[:, 0], 
        "rlnAngleTilt": angles[:, 1], 
        "rlnAnglePsi": angles[:, 2],
    })
    
    # 写入 STAR 文件（数据块名为 'data_0'）
    starfile.write({"data_0": df}, out_path, overwrite=True)


def _curve_to_der_points(curve: np.ndarray) -> np.ndarray:
    """由曲线点 (n, 3) 计算各点切向量，返回 (3, n) 的 der_points（ArtiaX CurvedLine 格式）。"""
    n = len(curve)
    tangents = np.zeros_like(curve)
    if n == 1:
        tangents[0] = 1.0
        return tangents.T
    tangents[0] = curve[1] - curve[0]
    tangents[-1] = curve[-1] - curve[-2]
    if n > 2:
        tangents[1:-1] = (curve[2:] - curve[:-2]) * 0.5
    return tangents.T  # (3, n)


def write_axm(path: Path, curve: np.ndarray, voxel_size: float, degree: int = 3, smooth: float = 0.0) -> Path:
    """
    写入 ArtiaX CurvedLine 格式的 .axm 文件，可在 ChimeraX+ArtiaX 中打开为曲线模型。
    
    格式与 ArtiaX CurvedLine.write_file 一致（内部为 npz）：model_type, particle_pos, degree,
    smooth, resolution, points (3×n), der_points (3×n)。通过 open(..., 'wb') 写入，保证文件后缀为 .axm
    而非 .npz。在 ArtiaX 中通过 Geometric Models → Open Geomodel... 选择本文件打开。
    
    参考: https://github.com/FrangakisLab/ArtiaX/blob/main/src/geometricmodel/CurvedLine.py
    """
    path = Path(path).with_suffix(".axm")
    # 修正：将像素坐标转换为埃 (Å) 坐标，以匹配 ArtiaX 的物理坐标系
    curve = np.asarray(curve * voxel_size, dtype=np.float32)
    n = len(curve)
    if n < 2:
        curve = np.vstack([curve[0], curve[0]])
        n = 2
    points = curve.T  # (3, n)，与 ArtiaX 一致
    der_points = _curve_to_der_points(curve)
    particle_pos = curve
    resolution = int(n)
    degree = int(degree)
    # 使用文件对象写入，避免 np.savez 将后缀改为 .npz
    with open(path, "wb") as f:
        np.savez(f, model_type="CurvedLine", particle_pos=particle_pos, degree=degree,
            smooth=smooth, resolution=resolution, points=points, der_points=der_points)
    return path


def interpolate_angles(t: np.ndarray, start_ang: np.ndarray, end_ang: np.ndarray) -> np.ndarray:
    """
    线性插值欧拉角（用于等间距模式的角度分配）。
    
    对于等间距采样的颗粒，其角度从纤维起始点的角度线性过渡到终止点的角度。
    这是一个简化假设，适用于角度变化不大的情况。
    
    注意：
    - 这是简单的线性插值，不考虑欧拉角的周期性（如 360° = 0°）
    - 若角度跨越 0°/360° 边界，可能产生不连续
    - 对于精确的角度插值，应使用四元数（quaternion）或旋转矩阵插值
    - 但对于大多数纤维应用，线性插值足够准确
    
    参数:
        t: 插值参数，形状为 (N,)，范围 [0, 1]
            - 0 对应起始点
            - 1 对应终止点
        start_ang: 起始角度 (rot, tilt, psi)，单位为度
        end_ang: 终止角度 (rot, tilt, psi)，单位为度
        
    返回:
        np.ndarray: 插值后的角度，形状为 (N, 3)
            - 每行为 (rot, tilt, psi)
            
    公式:
        angle(t) = start_ang + t * (end_ang - start_ang)
    """
    return start_ang + t[:, None] * (end_ang - start_ang)


# ========== 主函数（命令行入口）==========
def main():
    """
    主函数：解析命令行参数，执行纤维拟合和颗粒生成流程。
    
    工作流程：
    1. 解析命令行参数
    2. 读取 STAR 文件（纤维起止点）和 .rec 体积
    3. 对每条纤维进行卷积拟合
    4. 输出 .axm 文件（ArtiaX CurvedLine，可视化）
    5. 若指定 --out，生成颗粒 STAR（等间距或 MOD 过滤）
    6. 若在 ChimeraX 中运行，自动打开结果
    
    命令行参数详见 --help
    """
    parser = argparse.ArgumentParser(
        description="fiber start and end points + .rec → convolution/binarization/adjustable curvature fitting → particle .star or MOD filtering to .star",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # run in ChimeraX, generate curves
  runscript fiber_to_star.py --star fiber.star --rec volume.rec --voxel-size 14.5
  
  # equidistant mode: sample one particle every 40 Å along the curve
  python .\fiber_to_star.py --star .\data\cilia_106003_1_bin6.star --rec .\data\cilia_106003_1_bin6.rec --voxel-size 14.5 --radius 150 --spacing 40 --no-open
  
  # MOD filtering mode: filter particles near the curve from MOD
  python fiber_to_star.py --star fiber.star --rec volume.rec --voxel-size 14.5 --mod particles.mod --radius 200 --out filtered.star --no-open
  
  # adjust parameters for a single fiber
  python fiber_to_star.py --star fiber.star --rec volume.rec --voxel-size 14.5 --fiber-index 1 --curvature 0.2 --threshold 80 --erode 2
        """
    )
    
    # ===== 必选参数 =====
    parser.add_argument("--star", required=True, help="fiber start and end points .star file path (required)")
    parser.add_argument("--rec", required=True, help="input .rec volume path (required)")
    parser.add_argument("--voxel-size", type=float, required=True, help="voxel size (Å) (required, for coordinate scale conversion)")
    
    # ===== 纤维参数 =====
    parser.add_argument("--radius", type=float, default=150, help="fiber radius (Å), default 150, for convolution kernel and MOD filtering distance")
    parser.add_argument("--curvature", type=float, default=0.1, help="curvature [0,1], 0=straightest (strong smoothing), 1=tightest (follows skeleton), default 0.1")
    parser.add_argument("--threshold", type=float, default=75.0, help="binarization percentile (0-100), default 75, higher values are stricter")
    parser.add_argument("--erode", type=int, default=1, help="erode iterations after binarization, default 1, try 2-3 for discontinuous fibers")
    
    # ===== 输出模式（二选一）=====
    parser.add_argument("--spacing", type=float, default=40, help="equidistant sampling step (Å), default 40 (equidistant mode)")
    parser.add_argument("--mod", default=None, help="MOD file path: filter particles near the curve (MOD filtering mode)")
    parser.add_argument("--out", default=None, help="output .star file path (only when this parameter is specified, generate particle .star file)")
    
    # ===== 其他选项 =====
    # 卷积前预处理（归一化 + Gaussian + Kuwahara + Bottom-hat）
    parser.add_argument("--gaussian-sigma", type=float, default=None, metavar="F", help="pre-conv Gaussian sigma (px); default max(1, 0.5*radius_px)")
    parser.add_argument("--kuwahara-radius", type=int, default=3, metavar="N", help="3D Kuwahara window half-size (px), default 3")
    parser.add_argument("--bottomhat-radius", type=int, default=5, metavar="N", help="3D bottom-hat structure radius (px), default 5")
    parser.add_argument("--conv-mode", default="reflect", choices=("reflect", "nearest", "constant"), help="convolution border mode, default reflect")

    parser.add_argument("--curve-points", type=int, default=10, help="number of points per fitted curve (default 10, reduce for speed)")
    parser.add_argument("--fiber-index", type=int, default=None, metavar="N", help="if set, only fit and output fiber N (0-based); omit to process all fibers")
    parser.add_argument("--no-open", action="store_true", help="not automatically open results in ChimeraX (standalone mode)")
    parser.add_argument("--save-mrc", action="store_true", help="save intermediate MRC files for debugging")
    
    args = parser.parse_args()

    # ========== 参数验证与单位转换 ==========
    voxel = args.voxel_size
    if voxel <= 0:
        print("error: --voxel-size must be positive (Å).", file=sys.stderr)
        sys.exit(1)
    
    # 将半径和间距从 Å 转换为像素（体素单位）
    radius_px = args.radius / voxel
    spacing_px = args.spacing / voxel

    # ========== 路径处理 ==========
    # 转换为绝对路径（支持相对路径）
    star_path = Path(args.star)
    if not star_path.is_absolute():
        star_path = Path.cwd() / star_path
    
    rec_path = Path(args.rec)
    if not rec_path.is_absolute():
        rec_path = Path.cwd() / rec_path
    
    # 输出目录：与 STAR 文件同目录
    data_dir = star_path.parent
    
    # 验证 .rec 文件存在
    if not rec_path.exists():
        print(f"error: .rec file not found: {rec_path}", file=sys.stderr)
        sys.exit(1)

    # ========== 读取纤维起止点 ==========
    fibers, has_orphan = read_star_fibers(star_path)
    
    # 若 STAR 行数为奇数，提醒用户
    if has_orphan:
        print("warning: STAR file has odd number of lines, the last line (only start point) is ignored, the rest of the pairs are processed normally.", file=sys.stderr)
    
    # 验证至少有一条纤维
    if not fibers:
        print("error: no fiber start and end points pairs found (STAR file needs at least 2 lines).", file=sys.stderr)
        sys.exit(1)

    # 曲率参数限制在 [0, 1] 范围内
    curvature = np.clip(args.curvature, 0.0, 1.0)

    # ========== 确定要拟合的纤维（--fiber-index 功能）==========
    # 若指定 --fiber-index N 且 N 有效，仅拟合第 N 条纤维；否则拟合全部
    if args.fiber_index is not None and 0 <= args.fiber_index < len(fibers):
        indices_to_fit = [args.fiber_index]
        print(f"fiber-index={args.fiber_index}: fitting only fiber {args.fiber_index}")
    else:
        indices_to_fit = list(range(len(fibers)))
        if args.fiber_index is not None:
            print(f"warning: --fiber-index {args.fiber_index} out of range [0, {len(fibers)}), fitting all fibers.", file=sys.stderr)

    # ========== 加载体积 ==========
    vol = load_volume(rec_path)
    print(f"loaded volume shape: {vol.shape} (Z, Y, X)")
    
    # ========== 拟合纤维 ==========
    curves = []
    fibers_fitted = []  # 与 curves 一一对应的 (start_xyz, end_xyz, start_ang, end_ang)
    axm_paths = []
    for idx in indices_to_fit:
        start_xyz, end_xyz, start_ang, end_ang = fibers[idx]
        print(f"  fitting fiber {idx}: start {start_xyz} → end {end_xyz}")

        if args.save_mrc:
            debug_mrc_prefix = data_dir / f"{star_path.stem}_fiber{idx}"

        curve = fit_fiber_convolve(
            vol,
            start_xyz,
            end_xyz,
            radius_px=radius_px,
            threshold_percentile=args.threshold,
            curvature=curvature,
            erode_iterations=args.erode,
            curve_points=args.curve_points,
            debug_mrc_prefix=debug_mrc_prefix,
            voxel_size=args.voxel_size,
            gaussian_sigma=args.gaussian_sigma,
            kuwahara_radius=args.kuwahara_radius,
            bottomhat_radius=args.bottomhat_radius,
            convolution_mode=args.conv_mode,
        )
        curves.append(curve)
        fibers_fitted.append((start_xyz, end_xyz, start_ang, end_ang))
        axm_path = data_dir / f"{star_path.stem}_fiber{idx}.axm"
        write_axm(axm_path, curve, args.voxel_size)
        axm_paths.append(axm_path)
        print(f"    fitting completed, curve points: {len(curve)}, AXM file saved: {axm_path}")

    # ========== ChimeraX 集成（若在 ChimeraX 中运行）==========
    in_chimerax = _in_chimerax()
    if in_chimerax and not args.no_open and axm_paths:
        try:
            from chimerax.core.commands import run as chimerax_run
            session = _get_session()
            if session:
                for axm_path in axm_paths:
                    chimerax_run(session, f'open "{axm_path}"')
                session.logger.info(
                    f"opened {len(axm_paths)} .axm curve file(s) in ChimeraX.\n"
                    f"If the curve does not appear, use ArtiaX → Geometric Models → Open Geomodel... and select the .axm file.\n"
                    f"use --fiber-index N to refit only fiber N; use --out to export particles."
                )
        except Exception as e:
            print(f"warning: failed to open AXM in ChimeraX: {e}", file=sys.stderr)

    # ========== 生成颗粒 STAR（仅当指定 --out 时）==========
    # 当前拟合的纤维（可能为全部或 --fiber-index 指定的一条）都参与输出
    out_star = None
    if args.out:
        out_star = Path(args.out)
        all_coords = []  # 存储所有颗粒坐标
        all_angles = []  # 存储所有颗粒角度
        
        if args.mod:
            # ===== MOD 过滤模式 =====
            # 从 MOD 中读取手动挑选的颗粒，过滤出曲线附近的颗粒
            print(f"\nusing MOD filtering mode: {args.mod}")
            mod_path = Path(args.mod)
            if not mod_path.is_absolute():
                mod_path = Path.cwd() / mod_path
            
            # 读取 MOD 中的所有点
            pts = read_mod_points(mod_path)
            print(f"MOD has {len(pts)} points")
            
            # 对每条曲线，过滤出附近的点（距离 ≤ radius_px）
            for idx, (curve, (start_xyz, end_xyz, start_ang, end_ang)) in enumerate(zip(curves, fibers_fitted)):
                kept = filter_points_near_curve(pts, curve, radius_px)
                if len(kept) > 0:
                    all_coords.append(kept)
                    # MOD 无角度信息，使用纤维起始点角度（同一纤维内所有颗粒角度相同）
                    all_angles.append(np.tile(start_ang, (len(kept), 1)))
                    print(f"  fiber {indices_to_fit[idx]}: kept {len(kept)} points (distance ≤ {args.radius} Å)")
            
            # 合并所有纤维的颗粒
            if all_coords:
                all_coords = np.vstack(all_coords)
                all_angles = np.vstack(all_angles)
            else:
                print("warning: no MOD ·points near any curve, increase --radius or check if MOD matches the curves.", file=sys.stderr)
                all_coords = np.zeros((0, 3), dtype=np.float32)
                all_angles = np.zeros((0, 3), dtype=np.float32)
            
            # 写入 STAR 文件
            write_star(out_star, all_coords, all_angles)
            print(f"\nSTAR file saved: {out_star}, total particles: {len(all_coords)} from {len(curves)} fibers")
        
        else:
            # ===== 等间距模式 =====
            # 沿每条曲线按固定间距采样颗粒
            print(f"\nequidistant mode: spacing = {args.spacing} Å")
            
            # 对每条曲线进行等间距采样
            for idx, (curve, (start_xyz, end_xyz, start_ang, end_ang)) in enumerate(zip(curves, fibers_fitted)):
                # 重采样曲线（按 spacing_px 间距）
                sampled = resample_curve_by_spacing(curve, spacing_px)
                
                # 角度线性插值：从起始点角度过渡到终止点角度
                t = np.linspace(0, 1, len(sampled))
                angles = interpolate_angles(t, start_ang, end_ang)
                
                all_coords.append(sampled)
                all_angles.append(angles)
                print(f"  fiber {indices_to_fit[idx]}: sampled {len(sampled)} particles")
            
            # 合并所有纤维的颗粒
            all_coords = np.vstack(all_coords)
            all_angles = np.vstack(all_angles)
            
            # 写入 STAR 文件
            write_star(out_star, all_coords, all_angles)
            print(f"\nSTAR file saved: {out_star}, total particles: {len(all_coords)} from {len(curves)} fibers")

    # ========== 在 ChimeraX 中打开生成的 STAR（若适用）==========
    if in_chimerax and not args.no_open and out_star is not None and out_star.exists():
        try:
            from chimerax.core.commands import run as chimerax_run
            session = _get_session()
            if session:
                # 在 ChimeraX 中打开 STAR 文件（需要 ArtiaX 插件支持）
                chimerax_run(session, f'open "{out_star}" format star')
                session.logger.info(
                    f"particles STAR file opened in ChimeraX: {out_star}\n"
                    f"use ArtiaX's 'Open Particle List' function to view particles"
                )
        except Exception as e:
            print(f"warning: failed to open STAR in ChimeraX: {e}", file=sys.stderr)

    # ========== 完成信息 ==========
    if not in_chimerax:
        print("\nnot running in ChimeraX. use the following command to visualize the results in ChimeraX:")
        print(f"  runscript {Path(__file__).name} --star {star_path.name} --rec {rec_path.name} --voxel-size {args.voxel_size}")
    
    print("\nfinished!")


if __name__ == "__main__":
    main()
