#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emClarity BH_templateSearch3d.m 的 Python/CuPy GPU 复现
原始代码: https://github.com/StochasticAnalytics/emClarity/blob/main/alignment/BH_templateSearch3d.m

流程概述:
  1. 解析参数文件，初始化 GPU
  2. 加载断层图（tomogram）与模板（template）
  3. 构建角度搜索网格
  4. 预处理断层图（带通滤波 + 局部均值/RMS 归一化）
     - 默认：分块处理（内存友好）
     - --whole：整个断层图一次性上传 GPU，跳过分块
  5. 遍历所有旋转角度，对每块/整体做互相关，维护 MIP 和角度图
  6. 迭代式峰值拾取，输出 MRC/CSV/POS 结果文件

依赖: cupy, numpy, mrcfile, scipy, cupyx  (filament conda 环境)
"""

import argparse
import csv
import json
import os
import time

import cupy as cp
import cupyx.scipy.ndimage as cpnd
import mrcfile
import numpy as np
from scipy.ndimage import affine_transform
from scipy.ndimage import zoom as ndimage_zoom


# ===========================================================================
# 2. MRC 文件 I/O
# ===========================================================================

def load_mrc(path: str) -> np.ndarray:
    """
    读取 MRC 文件，返回 float32 numpy 数组。
    原始: BH_multi_loadOrBin / BH_multi_loadOrBuild
          [BH_templateSearch3d.m ~line 145, 150]
    """
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        return mrc.data.copy().astype(np.float32)


def save_mrc(data, path: str, voxel_size: float = 1.0):
    """保存 MRC 文件（自动从 GPU 拷回 CPU）。"""
    arr = cp.asnumpy(data).astype(np.float32) if isinstance(data, cp.ndarray) \
          else np.asarray(data, dtype=np.float32)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(arr)
        mrc.voxel_size = voxel_size


# ===========================================================================
# 3. 填充 / 裁剪工具
# ===========================================================================

def pad_val(size_in, size_out):
    """
    计算将 size_in 填充到 size_out 所需的前/后补零量。
    原始: BH_multi_padVal(size_in, size_out)
          [BH_templateSearch3d.m ~line 200, 297]
    返回: (pre, post) 均为长度 3 的 numpy int 数组
    """
    si   = np.array(size_in,  dtype=int)
    so   = np.array(size_out, dtype=int)
    diff = so - si
    pre  = diff // 2
    post = diff - pre
    return pre, post


def pad_zeros_3d(vol: cp.ndarray, pre, post) -> cp.ndarray:
    """
    GPU 数组零填充。
    原始: BH_padZeros3d(vol, pre, post, 'GPU', 'single')
          [BH_templateSearch3d.m 多处]
    """
    return cp.pad(vol, [(int(pre[i]), int(post[i])) for i in range(3)],
                  mode="constant")


# ===========================================================================
# 4. 带通滤波器（Fourier 空间）
# ===========================================================================

def bandpass_3d(size, low_res_ang: float, pixel_size_ang: float) -> cp.ndarray:
    """
    在 Fourier 空间构建三维带通滤波器，直接返回 GPU 数组。
    原始: BH_bandpass3d(sizeChunk, 10e-4, 800, lowResCut, 'cpu', pixelSize)
          [BH_templateSearch3d.m ~line 310]

    参数
    ----
    size          : (nx, ny, nz) 体积形状
    low_res_ang   : 低通截止分辨率（Å），即 lowResCut
    pixel_size_ang: 像素尺寸（Å）
    """
    nx, ny, nz = size
    fx = np.fft.fftfreq(nx).astype(np.float32)
    fy = np.fft.fftfreq(ny).astype(np.float32)
    fz = np.fft.fftfreq(nz).astype(np.float32)
    Fx, Fy, Fz = np.meshgrid(fx, fy, fz, indexing="ij")
    R_px = np.sqrt(Fx**2 + Fy**2 + Fz**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        R_ang = np.where(R_px > 0, pixel_size_ang / R_px, 1e9)

    # 低通：余弦软截止，过渡带 ±10%
    cutoff = low_res_ang
    edge   = cutoff * 0.1
    lp = np.where(
        R_ang > cutoff + edge, 0.0,
        np.where(R_ang < cutoff - edge, 1.0,
                 0.5 * (1 + np.cos(np.pi * (R_ang - (cutoff - edge)) / (2 * edge))))
    ).astype(np.float32)

    # 高通：抑制极低频（去梯度），截止约 800 Å
    hp_cutoff = 800.0
    hp = np.where(R_ang < hp_cutoff, 0.0,
                  np.where(R_ang > hp_cutoff * 1.1, 1.0,
                           (R_ang - hp_cutoff) / (hp_cutoff * 0.1))
                  ).astype(np.float32)

    return cp.asarray(lp * hp)


# ===========================================================================
# 5. 局部均值 / 局部 RMS（GPU 原生，cupyx.scipy.ndimage）
# ===========================================================================

def moving_average(vol_g: cp.ndarray, radius) -> cp.ndarray:
    """
    GPU 上计算局部均值（立方窗口）。
    原始: BH_movingAverage(tomoChunk, statsRadius)
          [BH_templateSearch3d.m ~line 370]
    使用 cupyx.scipy.ndimage.uniform_filter，全程留在 GPU。
    """
    r    = int(np.max(radius))
    size = 2 * r + 1
    return cpnd.uniform_filter(vol_g, size=size, mode="reflect")


def moving_rms(vol_g: cp.ndarray, radius) -> cp.ndarray:
    """
    GPU 上计算局部 RMS（立方窗口）。
    原始: BH_movingRMS(tomoChunk, statsRadius)
          [BH_templateSearch3d.m ~line 375]
    """
    r    = int(np.max(radius))
    size = 2 * r + 1
    return cp.sqrt(cp.maximum(
        cpnd.uniform_filter(vol_g**2, size=size, mode="reflect"), 1e-10
    ))


# ===========================================================================
# 6. 三维旋转（BAH Euler ZXZ 约定，CPU scipy）
# ===========================================================================

def euler_zxz_matrix(phi_deg: float, theta_deg: float,
                     psi_deg: float) -> np.ndarray:
    """
    由 Euler 角（ZXZ 约定，度）计算 3×3 旋转矩阵。
    emClarity 的 'Bah' 约定使用 ZXZ 顺序。
    原始: BH_defineMatrix([phi, theta, psi], 'Bah', 'inv')
          [BH_templateSearch3d.m ~line 580, 935]
    """
    phi   = np.radians(phi_deg)
    theta = np.radians(theta_deg)
    psi   = np.radians(psi_deg)

    cp_, sp_ = np.cos(phi),   np.sin(phi)
    ct_, st_ = np.cos(theta), np.sin(theta)
    cs_, ss_ = np.cos(psi),   np.sin(psi)

    # R = Rz(phi) · Rx(theta) · Rz(psi)
    return np.array([
        [ cp_*cs_ - sp_*ct_*ss_,  -cp_*ss_ - sp_*ct_*cs_,  sp_*st_],
        [ sp_*cs_ + cp_*ct_*ss_,  -sp_*ss_ + cp_*ct_*cs_, -cp_*st_],
        [ st_*ss_,                  st_*cs_,                 ct_    ],
    ], dtype=np.float64)


def rotate_volume(vol_np: np.ndarray,
                  phi: float, theta: float, psi: float) -> np.ndarray:
    """
    用 Euler 角旋转三维体积（CPU scipy affine_transform，线性插值）。
    原始: BH_resample3d(tempImg, [phi, theta, psi-phi], [1,1,1],
                        {'Bah',1,'linear',1}, 'GPU', 'forward')
          [BH_templateSearch3d.m ~line 540]
    注意: CuPy 尚无高阶样条体积旋转；模板通常较小，CPU 开销可接受。
    """
    R      = euler_zxz_matrix(phi, theta, psi)
    center = (np.array(vol_np.shape) - 1) / 2.0
    offset = center - R @ center
    return affine_transform(
        vol_np.astype(np.float32), R,
        offset=offset, order=1, mode="constant", cval=0.0
    ).astype(np.float32)


# ===========================================================================
# 7. 角度搜索网格
# ===========================================================================

def grid_search_angles(angle_search):
    """
    生成均匀球面角度搜索网格。
    原始: BH_multi_gridSearchAngles(angleSearch)
          [BH_templateSearch3d.m ~line 190]

    angle_search: [oop_range, oop_step, ip_range, ip_step]
    返回:
      n_in_plane    : 面内角度数
      in_plane_arr  : 面内角度数组（psi）
      angle_step    : (N, 3) 每行 = [n_phi_steps, theta, phi_step_deg]
      n_angles_total: 总角度数（整数）
    """
    if isinstance(angle_search, (list, np.ndarray)) and len(angle_search) >= 4:
        oop_range, oop_step, ip_range, ip_step = (float(x) for x in angle_search[:4])
    else:
        oop_range, oop_step, ip_range, ip_step = 90.0, 5.0, 180.0, 5.0

    in_plane_arr = np.arange(-ip_range, ip_range + ip_step * 0.5, ip_step)
    thetas       = np.arange(0, oop_range + oop_step * 0.5, oop_step)

    rows = []
    for theta in thetas:
        if theta < 1e-6:
            n_phi, phi_step = 1, 360.0
        else:
            n_phi     = max(1, int(np.round(360.0 * np.sin(np.radians(theta)) / oop_step)))
            phi_step  = 360.0 / n_phi
        rows.append([float(n_phi), float(theta), phi_step])

    angle_step     = np.array(rows, dtype=np.float32)
    n_in_plane     = len(in_plane_arr)
    n_angles_total = int(sum(int(r[0]) * n_in_plane for r in rows))
    return n_in_plane, in_plane_arr, angle_step, n_angles_total


# ===========================================================================
# 8. 分块迭代参数计算
# ===========================================================================

def multi_iterator(target_size, tomo_size, temp_size, lattice_rad2) -> dict:
    """
    计算分块处理所需的填充量、块大小、有效区域和迭代次数。
    原始: BH_multi_iterator([targetSize; size(tomogram); sizeTempBIN;
                             2*latticeRadius], 'convolution')
          [BH_templateSearch3d.m ~line 195]
    """
    ts = np.array(target_size,  dtype=int)
    to = np.array(tomo_size,    dtype=int)
    te = np.array(temp_size,    dtype=int)

    size_chunk = np.minimum(ts, to)
    valid_area = np.maximum(size_chunk - te, 1)
    n_iters    = np.ceil(to / valid_area).astype(int)
    tomo_pre   = np.zeros(3, dtype=int)
    padded     = n_iters * valid_area + (size_chunk - valid_area)
    tomo_post  = np.maximum(padded - to, 0).astype(int)

    return dict(tomo_pre=tomo_pre, tomo_post=tomo_post,
                size_chunk=size_chunk, valid_area=valid_area, n_iters=n_iters)


# ===========================================================================
# 9. 三维掩膜（球形 / 矩形）
# ===========================================================================

def mask_3d(mask_type: str, size, radius, center=None) -> np.ndarray:
    """
    生成三维球形或矩形软掩膜（CPU，需要时再上传 GPU）。
    原始: BH_mask3d(eraseMaskType, [2,2,2]*rmDim+1, eraseMaskRadius, [0,0,0])
          [BH_templateSearch3d.m ~line 810]
    """
    sz  = np.array(size, dtype=int)
    rad = np.array(radius, dtype=float)
    ctr = (sz - 1) / 2.0 if center is None else np.array(center, dtype=float)

    X, Y, Z = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
    X = X - ctr[0]; Y = Y - ctr[1]; Z = Z - ctr[2]

    if mask_type.lower() in ("sphere", "spherical"):
        R = np.sqrt((X / (rad[0] + 1e-10))**2 +
                    (Y / (rad[1] + 1e-10))**2 +
                    (Z / (rad[2] + 1e-10))**2)
        mask = 0.5 * (1 - np.tanh(np.pi * (R - 1.0) / 0.1))
        return np.clip(mask, 0, 1).astype(np.float32)
    else:
        mask = np.ones(sz, dtype=np.float32)
        mask[np.abs(X) > rad[0]] = 0
        mask[np.abs(Y) > rad[1]] = 0
        mask[np.abs(Z) > rad[2]] = 0
        return mask


# ===========================================================================
# 10. 缺失楔形掩膜（简化二值楔）
# ===========================================================================

def make_wedge_mask(size, tlt_data=None) -> np.ndarray:
    """
    生成 Fourier 空间缺失楔形掩膜（基于最小/最大倾斜角）。
    原始: BH_weightMask3d(-1.*OUTPUT(1,:), tiltGeometry, 'binaryWedgeGPU', ...)
          [BH_templateSearch3d.m ~line 215, ~line 415]
    """
    if tlt_data is None:
        return np.ones(size, dtype=np.float32)

    tilt_col   = tlt_data[:, 0] if tlt_data.ndim > 1 else tlt_data.ravel()
    min_t, max_t = np.min(tilt_col), np.max(tilt_col)

    nx, ny, nz = size
    Fx, Fy, Fz = np.meshgrid(np.fft.fftfreq(nx), np.fft.fftfreq(ny),
                               np.fft.fftfreq(nz), indexing="ij")
    Fxz      = np.sqrt(Fx**2 + Fz**2 + 1e-12)
    tilt_map = np.degrees(np.arctan2(Fy, Fxz))
    return ((tilt_map >= min_t) & (tilt_map <= max_t)).astype(np.float32)


# ===========================================================================
# 11. 带限、居中、归一化（模板 FFT 预处理，GPU）
# ===========================================================================

def band_limit_center_normalize(vol_np: np.ndarray,
                                 bandpass_g: cp.ndarray,
                                 pad_pre, pad_post) -> cp.ndarray:
    """
    填充 → 去均值 → FFT → 乘以带通，返回 GPU 复数 FFT。
    原始: BH_bandLimitCenterNormalize(tempRot, tempBandpass, '', padBIN, 'single')
          [BH_templateSearch3d.m ~line 565, 580]
    """
    pg = cp.asarray(vol_np)
    pg = pad_zeros_3d(pg, pad_pre, pad_post)
    pg = pg - cp.mean(pg)
    return cp.fft.fftn(pg) * bandpass_g


# ===========================================================================
# 12. 公共：单张 CCF 计算 + MIP 更新（inline 于主循环中）
# ===========================================================================

def _ccf_and_update(tomo_fou_g: cp.ndarray,
                    temp_fou_g: cp.ndarray,
                    trim_pre, trim_post,
                    mag_g: cp.ndarray, ang_g: cp.ndarray,
                    angle_idx: int):
    """
    计算一个角度的互相关，并原位更新 MIP（mag_g）和角度图（ang_g）。
    原始: ccfmap = fftshift(real(ifftn(tomoFou.*conj(tempFou))))
          ccfmap = ccfmap ./ std(ccfmap(:))
          replaceTmp = (magTmp < ccfmap)  [~line 592–665]
    """
    ccf = cp.fft.fftshift(cp.real(cp.fft.ifftn(tomo_fou_g * cp.conj(temp_fou_g))))

    # 裁剪出有效区域（whole 模式时 trim 为 0，等价于不裁剪）
    r0s = int(trim_pre[0]);  r0e = ccf.shape[0] - int(trim_post[0]) if trim_post[0] > 0 else ccf.shape[0]
    r1s = int(trim_pre[1]);  r1e = ccf.shape[1] - int(trim_post[1]) if trim_post[1] > 0 else ccf.shape[1]
    r2s = int(trim_pre[2]);  r2e = ccf.shape[2] - int(trim_post[2]) if trim_post[2] > 0 else ccf.shape[2]
    ccf = ccf[r0s:r0e, r1s:r1e, r2s:r2e]

    # 按 std 归一化
    # 原始: ccfmap = ccfmap ./ std(ccfmap(:)) [~line 598]
    std_v = float(cp.std(ccf))
    if std_v > 1e-10:
        ccf /= std_v

    replace = mag_g < ccf
    mag_g[replace] = ccf[replace]
    ang_g[replace] = float(angle_idx)


# ===========================================================================
# 13. 主函数：3D 模板搜索
# ===========================================================================

def template_search_3d(
    tomo_name: str,
    tomo_number: int,
    template_path: str,
    wedge_type: int,
    gpu_idx: int = 0,
    whole_tomo: bool = False,
):
    """Python/CuPy reimplement of BH_templateSearch3d.m main function."""
    t_start = time.time()

    # -----------------------------------------------------------------
    # GPU 初始化
    # 原始: gpuDevice(useGPU) [~line 25]
    # -----------------------------------------------------------------
    cp.cuda.Device(gpu_idx).use()
    dev_name = cp.cuda.runtime.getDeviceProperties(gpu_idx)["name"].decode()
    mode_str = "whole mode" if whole_tomo else "chunked mode"
    print(f"[GPU] device {gpu_idx}: {dev_name}  running mode: {mode_str}")

    # -----------------------------------------------------------------
    # 解析参数
    # 原始: pBH = BH_parseParameterFile(PARAMETER_FILE) [~line 40]
    # -----------------------------------------------------------------

    sampling_rate   = 6                                     # Tmp_samplingRate == binning factor line 49
    pixel_size_full = 2.417 * 1e10                          # m → Å  
    pixel_size = pixel_size_full * sampling_rate

    lattice_radius = np.array([420, 420, 420], dtype=float) # particleRadius line 77
    angle_search   = [180, 15, 180, 15]                     # Tmp_angleSearch
    target_size    = np.array([512, 512, 512], dtype=int)   # Tmp_targetSize line 81 default
    max_tries      = 10000                                  # max_peaks line 66 default
    n_peaks        = 1                                      # nPeaks ?
    wanted_cut     = 28.0                                   # lowResCut line 123 default
    peak_threshold = 3000                                   # Tmp_threshold line 74

    # 擦除掩膜参数
    # 原始: eraseMaskType, eraseMaskRadius [~line 95]
    erase_mask_type   = "sphere"                            # Peak_mType line 92 default
    erase_mask_radius = np.array([400, 400, 60], dtype=float) # Peak_mRadius line 95

    # -----------------------------------------------------------------
    # 加载 TLT 几何文件，计算 CTF 零点以决定低通截止
    # 原始: TLT = load(..._ctf.tlt); def = mean(-1.*TLT(:,15))*10^6;
    #        firstZero = -0.2*def^2 + 5.2*def + 11 [~line 128]
    # -----------------------------------------------------------------
    map_back_iter = 0
    tlt_path = f"fixedStacks/ctf/{tomo_name}_ali{map_back_iter+1}_ctf.tlt"
    tlt_data  = None
    first_zero = 0.0
    if os.path.exists(tlt_path):
        tlt_data = np.loadtxt(tlt_path)
        if tlt_data.ndim > 1 and tlt_data.shape[1] > 14:
            defocus_um = np.mean(-tlt_data[:, 14]) * 1e6
            first_zero = -0.2 * defocus_um**2 + 5.2 * defocus_um + 11.0

    # 原始: lowResCut = max(wantedCut, firstZero) [~line 128]
    low_res_cut = max(wanted_cut, first_zero)
    if pixel_size * 2 > low_res_cut:
        print(f"[分辨率] 限制到 Nyquist ({pixel_size*2:.2f} Å)")
        low_res_cut = pixel_size * 2
    else:
        print(f"[分辨率] lowResCut = {low_res_cut:.2f} Å "
              f"(wantedCut={wanted_cut}, firstZero={first_zero:.2f})")

    # -----------------------------------------------------------------
    # 加载断层图
    # 原始: BH_multi_loadOrBuild(...) [~line 145]
    # -----------------------------------------------------------------
    tomo_candidates = [
        f"cache/{tomo_name}_{tomo_number}_bin{sampling_rate}.rec",
        f"recon/{tomo_name}_{tomo_number}.rec",
        f"{tomo_name}_{tomo_number}.mrc",
        f"{tomo_name}_{tomo_number}.rec",
    ]
    tomo_path = next((p for p in tomo_candidates if os.path.exists(p)), None)
    if tomo_path is None:
        raise FileNotFoundError(f"找不到断层图，尝试路径: {tomo_candidates}")
    print(f"[加载] 断层图: {tomo_path}")
    tomogram       = load_mrc(tomo_path)
    tomo_size_orig = np.array(tomogram.shape)
    print(f"  断层图尺寸: {tomogram.shape}")

    # -----------------------------------------------------------------
    # 加载并预处理模板
    # 原始: BH_multi_loadOrBin(TEMPLATE, 1, 3) [~line 150]
    # -----------------------------------------------------------------
    print(f"[加载] 模板: {template_path}")
    template = load_mrc(template_path)
    print(f"  原始模板尺寸: {template.shape}")

    # 裁剪至 sqrt(2)*max(latticeRadius) 附近
    # 原始: trimTemp = BH_multi_padVal(size(template), ceil(2*max(Ali_mRadius./pixelSizeFULL))) [~line 155]
    ali_m_radius = np.array(pBH.get("Ali_mRadius", lattice_radius), dtype=float)
    trim_half = max(int(np.ceil(np.max(ali_m_radius / pixel_size_full))), 4)
    s = np.array(template.shape); c = s // 2; h = trim_half
    if np.all(c > h):
        template = template[c[0]-h:c[0]+h, c[1]-h:c[1]+h, c[2]-h:c[2]+h]
    print(f"  裁剪后模板尺寸: {template.shape}")

    # 保证偶数尺寸
    # 原始: template = padarray(template, mod(size(template),2), 0, 'post') [~line 165]
    pad_even = np.array(template.shape) % 2
    if np.any(pad_even):
        template = np.pad(template, [(0, int(p)) for p in pad_even])
    template = template - template.mean()

    # bin 到断层图采样率
    # 原始: templateBIN = BH_reScale3d(template,'',1/samplingRate,'cpu') [~line 167]
    if sampling_rate > 1:
        template_bin = ndimage_zoom(template, 1.0 / sampling_rate, order=1).astype(np.float32)
    else:
        template_bin = template.copy().astype(np.float32)

    template_bin -= template_bin.mean()
    rms_t = float(np.sqrt(np.mean(template_bin**2)))
    if rms_t > 0:
        template_bin /= rms_t

    size_temp_bin  = np.array(template_bin.shape)
    print(f"  BIN 后模板尺寸: {size_temp_bin}")

    # -----------------------------------------------------------------
    # 像素空间半径计算
    # 原始: latticeRadius = (0.75*latticeRadius)./pixelSize [~line 175]
    # -----------------------------------------------------------------
    stats_radius_px = np.ceil(
        2.0 * np.max(lattice_radius) / pixel_size
    ).astype(int) * np.ones(3, dtype=int)

    lattice_radius_px = np.floor(0.75 * lattice_radius / pixel_size).astype(int)
    lattice_radius_px += lattice_radius_px % 2

    erase_mask_radius_px = np.floor(erase_mask_radius / pixel_size).astype(int)
    erase_mask_radius_px += erase_mask_radius_px % 2

    print(f"  latticeRadius(px): {lattice_radius_px}")
    print(f"  eraseMaskRadius(px): {erase_mask_radius_px}")
    print(f"  statsRadius(px): {stats_radius_px}")

    # -----------------------------------------------------------------
    # 角度搜索网格
    # 原始: BH_multi_gridSearchAngles(angleSearch) [~line 190]
    # -----------------------------------------------------------------
    n_in_plane, in_plane_arr, angle_step, n_angles_total = grid_search_angles(
        angle_search
    )
    print(f"[角度] 总角度数: {n_angles_total}，面内角度数: {n_in_plane}")

    # -----------------------------------------------------------------
    # 输出目录
    # 原始: convTMPNAME = sprintf('convmap_wedgeType_%d_bin%d',...) [~line 100]
    # -----------------------------------------------------------------
    conv_name = f"convmap_wedgeType_{wedge_type}_bin{sampling_rate}"
    os.makedirs(conv_name, exist_ok=True)
    map_name  = f"{tomo_name}_{tomo_number}_bin{sampling_rate}"

    ANGLE_LIST = np.zeros((n_angles_total, 3), dtype=np.float32)

    # =================================================================
    # ===  执行路径 A：整体模式（--whole）                           ===
    # =================================================================
    if whole_tomo:
        RESULTS_peak, RESULTS_angle, current_global_angle = _run_whole(
            tomogram, tomo_size_orig, template_bin, size_temp_bin,
            tlt_data, low_res_cut, pixel_size, stats_radius_px,
            angle_step, in_plane_arr, n_angles_total, n_in_plane,
            ANGLE_LIST,
        )
    # =================================================================
    # ===  执行路径 B：分块模式（默认）                              ===
    # =================================================================
    else:
        RESULTS_peak, RESULTS_angle, current_global_angle = _run_chunked(
            tomogram, tomo_size_orig, template_bin, size_temp_bin,
            tlt_data, low_res_cut, pixel_size, stats_radius_px,
            target_size, lattice_radius_px,
            angle_step, in_plane_arr, n_angles_total, n_in_plane,
            ANGLE_LIST,
        )

    # -----------------------------------------------------------------
    # 保存互相关图 / 角度图 / 角度列表
    # 原始: SAVE_IMG(MRCImage(mag), resultsOUT) [~line 760]
    # -----------------------------------------------------------------
    save_mrc(RESULTS_peak,  f"{conv_name}/{map_name}_convmap.mrc", voxel_size=pixel_size)
    save_mrc(RESULTS_angle, f"{conv_name}/{map_name}_angles.mrc",  voxel_size=pixel_size)
    np.savetxt(f"{conv_name}/{map_name}_angles.list",
               ANGLE_LIST[:current_global_angle], fmt="%.2f\t%.2f\t%.2f")
    print(f"\n[输出] 互相关图: {conv_name}/{map_name}_convmap.mrc")
    print(f"[输出] 角度图:   {conv_name}/{map_name}_angles.mrc")

    # -----------------------------------------------------------------
    # 峰值拾取（分块 / 整体模式共用）
    # 原始: Tmean + peakThreshold*Tstd 阈值化，while 循环迭代拾取 [~line 790]
    # -----------------------------------------------------------------
    peak_mat = _pick_peaks(
        RESULTS_peak, RESULTS_angle, ANGLE_LIST,
        lattice_radius_px, erase_mask_radius_px, erase_mask_type,
        peak_threshold, max_tries, n_peaks, current_global_angle,
        sampling_rate,
    )

    # -----------------------------------------------------------------
    # 写出 CSV / POS / PATH 文件
    # 原始: csv_out, pos_out file writing [~line 910]
    # -----------------------------------------------------------------
    csv_out  = f"{conv_name}/{map_name}.csv"
    pos_out  = f"{conv_name}/{map_name}.pos"
    path_out = f"{conv_name}/{map_name}.path"

    with open(csv_out, "w", newline="") as f_csv, open(pos_out, "w") as f_pos:
        writer = csv.writer(f_csv, delimiter=" ")
        n_out = 0
        for pk in peak_mat:
            if pk["x"] <= 0 or pk["y"] <= 0 or pk["z"] <= 0:
                continue
            # 原始: r = reshape(BH_defineMatrix(peakMat(i,4:6),'Bah','inv'),1,9) [~line 935]
            R_inv = euler_zxz_matrix(pk["phi"], pk["theta"], pk["psi"]).T.ravel()
            row = (
                [f"{pk['score']:.2f}", sampling_rate, 0,
                 n_out + 1, 1, 1, 1, 1, 1, 0,
                 f"{pk['x']:.4f}", f"{pk['y']:.4f}", f"{pk['z']:.4f}",
                 f"{pk['phi']:.2f}", f"{pk['theta']:.2f}", f"{pk['psi']:.2f}"]
                + [f"{v:.6f}" for v in R_inv] + [1]
            )
            writer.writerow(row)
            f_pos.write(f"{pk['x']/sampling_rate:.4f} "
                        f"{pk['y']/sampling_rate:.4f} "
                        f"{pk['z']/sampling_rate:.4f}\n")
            n_out += 1

    # 路径文件供 emClarity 后续步骤使用
    # 原始: fprintf(fileID,'%s,%s,%s,%s', mapName, mapPath, mapExt, RAWTLT) [~line 955]
    with open(path_out, "w") as fh:
        fh.write(f"{map_name},./cache,.rec,"
                 f"fixedStacks/ctf/{tomo_name}_ali1_ctf.tlt")

    print(f"[输出] CSV:  {csv_out}  ({n_out} 个粒子)")
    print(f"[输出] POS:  {pos_out}")
    print(f"\n总执行时间: {time.time()-t_start:.1f} 秒")

    return RESULTS_peak, RESULTS_angle


# ===========================================================================
# 14. 执行路径 A：整体断层图模式
# ===========================================================================

def _run_whole(
    tomogram, tomo_size_orig, template_bin, size_temp_bin,
    tlt_data, low_res_cut, pixel_size, stats_radius_px,
    angle_step, in_plane_arr, n_angles_total, n_in_plane,
    ANGLE_LIST,
):
    """
    将整个断层图一次性上传 GPU，完成预处理和全部角度匹配。
    无分块，FFT 仅在每个面外角度（theta）变化时重新计算模板 FFT，
    断层图 FFT 仅计算一次。
    """
    print("\n[整体模式] 上传断层图到 GPU...")
    tomo_g = cp.asarray(tomogram)

    # 带通滤波器（基于完整断层图尺寸）
    # 原始: BH_bandpass3d(sizeChunk, ...) [~line 310] — 此处 chunk = 整个断层图
    bandpass_g = bandpass_3d(tuple(tomo_size_orig), low_res_cut, pixel_size)

    # 带通滤波
    # 原始: tomoChunk = real(ifftn(fftn(tomoChunk).*tomoBandpass)) [~line 355]
    print("  带通滤波...")
    tomo_g = cp.real(cp.fft.ifftn(cp.fft.fftn(tomo_g) * bandpass_g))

    # 局部均值 / RMS 归一化（全图）
    # 原始: averageMask + rmsMask [~line 370–390]
    print("  局部归一化（moving avg/rms）...")
    avg = moving_average(tomo_g, stats_radius_px)
    tomo_g = tomo_g - avg
    rms = cp.maximum(moving_rms(tomo_g, stats_radius_px), 1e-6)
    tomo_g = tomo_g / rms

    # 全局方差归一化
    # 原始: globalVariance = (fullX2 - fullX)/fullnX [~line 405]
    global_var = float((cp.sum(tomo_g**2) - cp.sum(tomo_g)) / tomo_g.size)
    if global_var > 0:
        tomo_g /= global_var
    print(f"  全局方差因子: {global_var:.4f}")

    # 计算断层图 FFT（只需一次）
    # 原始: tomoFou = fftn(gpuArray(tomoStack(:,:,:,tomoIDX))) [~line 510]
    print("  计算断层图 FFT...")
    tomo_fou_g = cp.fft.fftn(tomo_g)
    del tomo_g; cp.get_default_memory_pool().free_all_blocks()

    # 模板到断层图尺寸的填充量
    pad_pre, pad_post = pad_val(size_temp_bin, tomo_size_orig)
    # whole 模式下 CCF 与断层图等大，trim 为零
    trim_pre  = np.zeros(3, dtype=int)
    trim_post = np.zeros(3, dtype=int)

    # 结果体积（GPU）
    # 原始: RESULTS_peak = zeros(sizeTomo,'single') [~line 300]
    RESULTS_peak_g  = cp.zeros(tomo_size_orig, dtype=cp.float32)
    RESULTS_angle_g = cp.zeros(tomo_size_orig, dtype=cp.float32)

    interp_norm       = float(np.sum(template_bin**2))
    current_global_angle = 0
    reference_stack      = {}   # 缓存旋转后的模板（phi 变化才重算）
    n_complete = 0
    t_loop = time.time()

    print("\n[整体模式] 开始模板匹配...")
    for iAngle in range(len(angle_step)):
        theta        = float(angle_step[iAngle, 1])
        phi_step_val = float(angle_step[iAngle, 2])
        n_phi_steps  = int(angle_step[iAngle, 0])

        # ----- 方位角 phi 循环 -----
        # 原始: for iAzimuth = 0:angleStep(iAngle,2) [~line 515]
        for iAzimuth in range(n_phi_steps + 1):
            phi = phi_step_val * iAzimuth

            # ----- 面内角 psi 循环 -----
            # 原始: for iInPlane = inPlaneSearch [~line 520]
            for psi in in_plane_arr:
                angles = [phi, theta, float(psi) - phi]
                if current_global_angle < n_angles_total:
                    ANGLE_LIST[current_global_angle] = angles

                # 旋转模板（CPU），结果缓存
                # 原始: BH_resample3d(tempImg, [phi,theta,psi-phi], ...) [~line 540]
                key = (round(phi, 4), round(theta, 4), round(float(psi)-phi, 4))
                if key not in reference_stack:
                    temp_rot = rotate_volume(template_bin, *angles)
                    rot_norm = float(np.sum(temp_rot**2))
                    if rot_norm > 1e-10:
                        temp_rot *= interp_norm / rot_norm
                    reference_stack[key] = temp_rot
                else:
                    temp_rot = reference_stack[key]

                # 带限 + FFT（填充到断层图大小）
                # 原始: BH_bandLimitCenterNormalize(...) [~line 565]
                temp_fou_g = band_limit_center_normalize(
                    temp_rot, bandpass_g, pad_pre, pad_post
                )

                # CCF + MIP 更新
                _ccf_and_update(tomo_fou_g, temp_fou_g,
                                trim_pre, trim_post,
                                RESULTS_peak_g, RESULTS_angle_g,
                                current_global_angle + 1)

                current_global_angle += 1
                n_complete += 1

        if (iAngle + 1) % max(1, len(angle_step) // 10) == 0:
            elapsed = time.time() - t_loop
            rate    = n_complete / elapsed
            remain  = (n_angles_total - n_complete) / rate if rate > 0 else 0
            print(f"  [{iAngle+1}/{len(angle_step)}] θ={theta:.1f}°  "
                  f"{n_complete}/{n_angles_total} 角度  "
                  f"速率 {rate:.0f}/s  剩余 {remain:.0f}s")

    RESULTS_peak  = cp.asnumpy(RESULTS_peak_g)
    RESULTS_angle = cp.asnumpy(RESULTS_angle_g)
    return RESULTS_peak, RESULTS_angle, current_global_angle


# ===========================================================================
# 15. 执行路径 B：分块模式（默认）
# ===========================================================================

def _run_chunked(
    tomogram, tomo_size_orig, template_bin, size_temp_bin,
    tlt_data, low_res_cut, pixel_size, stats_radius_px,
    target_size, lattice_radius_px,
    angle_step, in_plane_arr, n_angles_total, n_in_plane,
    ANGLE_LIST,
):
    """
    分块预处理断层图，然后分块 × 角度双层循环计算互相关。
    原始逻辑的直接对应版本，全程使用 CuPy GPU。
    """
    # 分块参数
    # 原始: BH_multi_iterator([targetSize; size(tomogram); sizeTempBIN;
    #        2*latticeRadius], 'convolution') [~line 195]
    it = multi_iterator(target_size, tomo_size_orig, size_temp_bin,
                        2 * lattice_radius_px)
    tomo_pre   = it["tomo_pre"]
    tomo_post  = it["tomo_post"]
    size_chunk = it["size_chunk"]
    valid_area = it["valid_area"]
    n_iters    = it["n_iters"]
    print(f"[分块] 预填充={tomo_pre}  后填充={tomo_post}")
    print(f"       块大小={size_chunk}  有效区域={valid_area}  迭代次数={n_iters}")

    # 带通滤波器（块空间尺寸）
    # 原始: BH_bandpass3d(sizeChunk, ...) [~line 310]
    bandpass_g = bandpass_3d(tuple(size_chunk), low_res_cut, pixel_size)

    # 有效区域矩形掩膜（CPU，后面 broadcast 用）
    # 原始: BH_mask3d('rectangle', sizeChunk, validCalc./2, [0,0,0]) [~line 295]
    valid_area_mask_g = cp.asarray(
        mask_3d("rectangle", size_chunk, valid_area / 2)
    )

    # 裁剪 / 填充量
    vA_pre, vA_post           = pad_val(valid_area, size_chunk)
    pad_bin_pre, pad_bin_post = pad_val(size_temp_bin, size_chunk)
    trim_pre, trim_post       = pad_val(valid_area, size_chunk)  # 同 vA

    # 对称填充断层图
    # 原始: tomogram = padarray(tomogram, tomoPre, 'symmetric', ...) [~line 290]
    tomogram = np.pad(tomogram,
                      [(tomo_pre[i], tomo_post[i]) for i in range(3)],
                      mode="symmetric")
    size_tomo = np.array(tomogram.shape)
    print(f"[填充] 断层图填充后尺寸: {size_tomo}")

    # -----------------------------------------------------------------
    # 预处理所有断层图块
    # 原始: for iX = 1:nIters(1) ... [~line 330]
    # -----------------------------------------------------------------
    n_tomograms = int(np.prod(n_iters))
    tomo_stack  = np.zeros(list(size_chunk) + [n_tomograms], dtype=np.float32)
    tomo_coords = np.zeros((n_tomograms, 3), dtype=int)

    full_x = 0.0; full_x2 = 0.0; full_nx = 0
    tomo_idx = 0

    print("\n[预处理] 开始断层图分块预处理...")
    for iX in range(n_iters[0]):
        cut_x = iX * valid_area[0]
        for iY in range(n_iters[1]):
            cut_y = iY * valid_area[1]
            for iZ in range(n_iters[2]):
                cut_z = iZ * valid_area[2]
                print(f"  块 [{iX+1}/{n_iters[0]}, "
                      f"{iY+1}/{n_iters[1]}, {iZ+1}/{n_iters[2]}]", end=" ... ")

                # 提取块并上传 GPU
                # 原始: tomoChunk = tomogram(cutX:...) [~line 345]
                chunk_g = cp.asarray(tomogram[
                    cut_x:cut_x+size_chunk[0],
                    cut_y:cut_y+size_chunk[1],
                    cut_z:cut_z+size_chunk[2]
                ])

                # 带通滤波
                # 原始: tomoChunk = real(ifftn(fftn(tomoChunk).*tomoBandpass)) [~line 355]
                chunk_g = cp.real(cp.fft.ifftn(cp.fft.fftn(chunk_g) * bandpass_g))

                # 局部均值 / RMS 归一化（全程 GPU）
                # 原始: BH_movingAverage / BH_movingRMS [~line 370–380]
                avg_g   = moving_average(chunk_g, stats_radius_px)
                chunk_g = chunk_g - avg_g
                rms_g   = cp.maximum(moving_rms(chunk_g, stats_radius_px), 1e-6)

                # 乘以有效区域掩膜，写回 CPU stack
                # 原始: tomoStack(:,:,:,tomoIDX) = gather((tomoChunk./rmsMask).*validAreaMask) [~line 390]
                norm_g = (chunk_g / rms_g) * valid_area_mask_g
                normalized = cp.asnumpy(norm_g)

                tomo_stack[:, :, :, tomo_idx] = normalized
                full_x  += float(cp.sum(norm_g))
                full_x2 += float(cp.sum(norm_g**2))
                full_nx += int(norm_g.size)

                tomo_coords[tomo_idx] = [cut_x, cut_y, cut_z]
                tomo_idx += 1
                print("完成")

    # 全局方差归一化
    # 原始: globalVariance = (fullX2 - fullX)/fullnX [~line 405]
    global_var = (full_x2 - full_x) / max(full_nx, 1)
    if global_var > 0:
        tomo_stack /= global_var
    print(f"  全局方差归一化因子: {global_var:.4f}")

    # 结果体积（CPU，在 GPU 上更新后写回）
    # 原始: RESULTS_peak = zeros(sizeTomo,'single') [~line 300]
    RESULTS_peak  = np.zeros(size_tomo, dtype=np.float32)
    RESULTS_angle = np.zeros(size_tomo, dtype=np.float32)

    interp_norm          = float(np.sum(template_bin**2))
    current_global_angle = 0
    n_complete           = 0
    total_time_match     = 0.0
    first_loop_over_tomo = True

    # -----------------------------------------------------------------
    # 模板匹配主循环
    # 原始: for iAngle = 1:size(angleStep,1) ... [~line 450]
    # -----------------------------------------------------------------
    print("\n[分块匹配] 开始模板匹配主循环...")
    for iAngle in range(len(angle_step)):
        theta        = float(angle_step[iAngle, 1])
        phi_step_val = float(angle_step[iAngle, 2])
        n_phi_steps  = int(angle_step[iAngle, 0])

        n_ref_iter = n_phi_steps * n_in_plane + 1
        # 预计算该面外角度下所有旋转参考模板
        # 原始: referenceStack = zeros([sizeTempBIN, numRefIter], ...) [~line 480]
        reference_stack = np.zeros(list(size_temp_bin) + [n_ref_iter], dtype=np.float32)

        tomo_idx_inner    = 0
        first_loop_over_angle = True

        for iTomo in range(n_tomograms):
            t_chunk = time.time()
            i_cut   = tomo_coords[tomo_idx_inner]

            intra_loop_angle  = 0
            first_loop_over_chunk = True

            # FFT 当前断层图块（GPU）
            # 原始: tomoFou = fftn(gpuArray(tomoStack(:,:,:,tomoIDX))) [~line 510]
            tomo_fou_g = cp.fft.fftn(cp.asarray(tomo_stack[:, :, :, tomo_idx_inner]))

            mag_g: cp.ndarray | None = None
            ang_g: cp.ndarray | None = None

            # ----- 方位角 phi 循环 -----
            # 原始: for iAzimuth = 0:angleStep(iAngle,2) [~line 515]
            for iAzimuth in range(n_phi_steps + 1):
                phi = phi_step_val * iAzimuth

                # ----- 面内角 psi 循环 -----
                # 原始: for iInPlane = inPlaneSearch [~line 520]
                for psi in in_plane_arr:
                    angles = [phi, theta, float(psi) - phi]
                    if current_global_angle < n_angles_total:
                        ANGLE_LIST[current_global_angle] = angles

                    # 旋转模板（CPU，仅第一个块时计算，后续复用）
                    # 原始: if firstLoopOverAngle → BH_resample3d [~line 535]
                    if first_loop_over_angle:
                        temp_rot = rotate_volume(template_bin, *angles)
                        rot_norm = float(np.sum(temp_rot**2))
                        if rot_norm > 1e-10:
                            temp_rot *= interp_norm / rot_norm
                        reference_stack[:, :, :, intra_loop_angle] = temp_rot
                    else:
                        temp_rot = reference_stack[:, :, :, intra_loop_angle]

                    # 带限 + FFT（GPU）
                    # 原始: BH_bandLimitCenterNormalize(...) [~line 565, 578]
                    temp_fou_g = band_limit_center_normalize(
                        temp_rot, bandpass_g, pad_bin_pre, pad_bin_post
                    )

                    # 第一块第一角度：初始化 MIP
                    # 原始: firstLoopOverTomo && firstLoopOverChunk [~line 620]
                    if first_loop_over_tomo and first_loop_over_chunk:
                        ccf = cp.fft.fftshift(cp.real(cp.fft.ifftn(
                            tomo_fou_g * cp.conj(temp_fou_g)
                        )))
                        r0s = int(trim_pre[0]); r0e = ccf.shape[0] - int(trim_post[0]) if trim_post[0] > 0 else ccf.shape[0]
                        r1s = int(trim_pre[1]); r1e = ccf.shape[1] - int(trim_post[1]) if trim_post[1] > 0 else ccf.shape[1]
                        r2s = int(trim_pre[2]); r2e = ccf.shape[2] - int(trim_post[2]) if trim_post[2] > 0 else ccf.shape[2]
                        ccf = ccf[r0s:r0e, r1s:r1e, r2s:r2e]
                        std_v = float(cp.std(ccf))
                        if std_v > 1e-10:
                            ccf /= std_v
                        mag_g = ccf.copy()
                        ang_g = cp.ones(ccf.shape, dtype=cp.float32)
                        first_loop_over_tomo  = False
                        first_loop_over_chunk = False

                    elif first_loop_over_chunk:
                        # 从全局结果中取出该块的已有 MIP
                        # 原始: elseif firstLoopOverChunk [~line 638]
                        s0 = int(i_cut[0]); e0 = s0 + int(size_chunk[0])
                        s1 = int(i_cut[1]); e1 = s1 + int(size_chunk[1])
                        s2 = int(i_cut[2]); e2 = s2 + int(size_chunk[2])
                        mag_store = RESULTS_peak [s0:e0, s1:e1, s2:s2+size_chunk[2]]
                        ang_store = RESULTS_angle[s0:e0, s1:e1, s2:s2+size_chunk[2]]
                        # 裁剪出有效区域
                        va0s = int(vA_pre[0]); va0e = mag_store.shape[0] - int(vA_post[0]) if vA_post[0] > 0 else mag_store.shape[0]
                        va1s = int(vA_pre[1]); va1e = mag_store.shape[1] - int(vA_post[1]) if vA_post[1] > 0 else mag_store.shape[1]
                        va2s = int(vA_pre[2]); va2e = mag_store.shape[2] - int(vA_post[2]) if vA_post[2] > 0 else mag_store.shape[2]
                        mag_g = cp.asarray(mag_store[va0s:va0e, va1s:va1e, va2s:va2e])
                        ang_g = cp.asarray(ang_store[va0s:va0e, va1s:va1e, va2s:va2e])
                        first_loop_over_chunk = False

                        _ccf_and_update(tomo_fou_g, temp_fou_g,
                                        trim_pre, trim_post,
                                        mag_g, ang_g, current_global_angle + 1)
                    else:
                        # 常规更新
                        # 原始: replaceTmp = (magTmp < ccfmap) [~line 665]
                        _ccf_and_update(tomo_fou_g, temp_fou_g,
                                        trim_pre, trim_post,
                                        mag_g, ang_g, current_global_angle + 1)

                    intra_loop_angle     += 1
                    current_global_angle += 1
                    n_complete           += 1

            # 将该块 MIP 写回全局体积
            # 原始: RESULTS_peak(iCut:...) = magStoreTmp [~line 695]
            s0 = int(i_cut[0]); e0 = s0 + int(size_chunk[0])
            s1 = int(i_cut[1]); e1 = s1 + int(size_chunk[1])
            s2 = int(i_cut[2]); e2 = s2 + int(size_chunk[2])

            ws0 = s0 + int(vA_pre[0]); we0 = e0 - int(vA_post[0]) if vA_post[0] > 0 else e0
            ws1 = s1 + int(vA_pre[1]); we1 = e1 - int(vA_post[1]) if vA_post[1] > 0 else e1
            ws2 = s2 + int(vA_pre[2]); we2 = e2 - int(vA_post[2]) if vA_post[2] > 0 else e2

            RESULTS_peak [ws0:we0, ws1:we1, ws2:we2] = cp.asnumpy(mag_g)
            RESULTS_angle[ws0:we0, ws1:we1, ws2:we2] = cp.asnumpy(ang_g)

            elapsed_chunk = time.time() - t_chunk
            total_time_match += elapsed_chunk
            rate = n_complete / total_time_match if total_time_match > 0 else 0
            remain = (n_angles_total * n_tomograms - n_complete) / rate if rate > 0 else 0
            print(f"  角度[{iAngle+1}/{len(angle_step)}] "
                  f"块[{tomo_idx_inner+1}/{n_tomograms}]  "
                  f"{elapsed_chunk:.1f}s  剩余估计 {remain:.0f}s")

            tomo_idx_inner       += 1
            first_loop_over_angle = False
            current_global_angle -= intra_loop_angle

        current_global_angle += intra_loop_angle

    # 裁剪掉分块填充
    # 原始: RESULTS_peak = RESULTS_peak(1+tomoPre(1):end-tomoPost(1), ...) [~line 740]
    def _trim(vol):
        s = vol.shape
        return vol[
            tomo_pre[0]: s[0]-tomo_post[0] if tomo_post[0] > 0 else s[0],
            tomo_pre[1]: s[1]-tomo_post[1] if tomo_post[1] > 0 else s[1],
            tomo_pre[2]: s[2]-tomo_post[2] if tomo_post[2] > 0 else s[2],
        ]
    RESULTS_peak  = _trim(RESULTS_peak)
    RESULTS_angle = _trim(RESULTS_angle)
    return RESULTS_peak, RESULTS_angle, current_global_angle


# ===========================================================================
# 16. 峰值拾取（分块/整体模式共用）
# ===========================================================================

def _pick_peaks(
    RESULTS_peak, RESULTS_angle, ANGLE_LIST,
    lattice_radius_px, erase_mask_radius_px, erase_mask_type,
    peak_threshold, max_tries, n_peaks, current_global_angle,
    sampling_rate,
) -> list:
    """
    迭代式峰值拾取：统计阈值化 → 找最大 → 中心质量细化 → 旋转擦除 → 循环。
    原始: while n <= peakThreshold && this_try < max_tries [~line 820]
    全程在 GPU 上执行 argmax / 擦除，CPU 只做少量中心质量计算。
    """
    szK    = lattice_radius_px
    rm_dim = np.maximum(np.max(erase_mask_radius_px), np.max(szK)) * np.ones(3, dtype=int)

    # 边缘裁剪 + 再填充
    # 原始: mag = mag(szK(1)+1:end-szK(1),...) + BH_padZeros3d(...) [~line 800]
    mag = RESULTS_peak.copy()
    mag_inner = mag[szK[0]: mag.shape[0]-szK[0],
                    szK[1]: mag.shape[1]-szK[1],
                    szK[2]: mag.shape[2]-szK[2]]
    mag = np.pad(mag_inner,
                 [(int(szK[i]+rm_dim[i]), int(szK[i]+rm_dim[i])) for i in range(3)])
    Ang = np.pad(RESULTS_angle,
                 [(int(rm_dim[i]), int(rm_dim[i])) for i in range(3)])

    nz   = mag != 0
    T_mean = float(np.mean(mag[nz])) if np.any(nz) else 0.0
    T_std  = float(np.std(mag[nz]))  if np.any(nz) else 1.0

    # 原始: mag((Ang < 0)) = 0 [~line 802]
    mag[Ang < 0] = 0.0

    print(f"\n[峰值] mean={T_mean:.4f}  std={T_std:.4f}  "
          f"阈值(mean+{peak_threshold:.1f}*std)={T_mean+peak_threshold*T_std:.4f}")

    # 上传到 GPU
    # 原始: mag = gpuArray(mag) [~line 803]
    mag_g = cp.asarray(mag)
    Ang_g = cp.asarray(Ang)
    size_pick = np.array(mag.shape)

    # 擦除掩膜
    # 原始: removalMask = BH_mask3d(eraseMaskType, [2,2,2]*rmDim+1, ...) [~line 810]
    erase_size   = (2 * rm_dim + 1).astype(int)
    removal_mask = mask_3d(erase_mask_type, erase_size, erase_mask_radius_px)
    mask_cut_off = 0.999

    max_val = float(cp.max(mag_g))
    coord   = int(cp.argmax(mag_g))

    peak_mat = []
    n_found  = 0
    this_try = 0

    # ----- 迭代峰值拾取 while 循环 -----
    # 原始: while n <= peakThreshold && this_try < max_tries [~line 820]
    while n_found < int(peak_threshold) and this_try < max_tries:
        this_try += 1
        i, j, k = np.unravel_index(coord, tuple(size_pick))

        if float(Ang_g[i, j, k]) > 0:
            # 中心质量细化
            # 原始: bDist, cMass = sum(magBox.*cmX)/sum(magBox) [~line 835]
            b_dist = 1 + round(float(np.log(max(n_peaks, 1) + 1)))
            cl = [max(i-b_dist, 0), max(j-b_dist, 0), max(k-b_dist, 0)]
            ch = [min(i+b_dist, size_pick[0]-1),
                  min(j+b_dist, size_pick[1]-1),
                  min(k+b_dist, size_pick[2]-1)]

            mag_box = cp.asnumpy(mag_g[cl[0]:ch[0]+1, cl[1]:ch[1]+1, cl[2]:ch[2]+1])
            ang_box = cp.asnumpy(Ang_g[cl[0]:ch[0]+1, cl[1]:ch[1]+1, cl[2]:ch[2]+1])
            bx, by, bz = mag_box.shape

            cx_, cy_, cz_ = np.mgrid[-b_dist:b_dist+1,
                                     -b_dist:b_dist+1,
                                     -b_dist:b_dist+1]
            cx_ = cx_[:bx, :by, :bz]; cy_ = cy_[:bx, :by, :bz]; cz_ = cz_[:bx, :by, :bz]
            tm  = float(np.sum(mag_box))
            c_mass = (np.array([np.sum(mag_box * cx_), np.sum(mag_box * cy_),
                                np.sum(mag_box * cz_)]) / tm
                      if tm > 0 else np.zeros(3))

            # 位置（减去填充）
            # 原始: cenP = c + cMass' - rmDim [~line 860]
            cen_p = np.array([i, j, k], dtype=float) + c_mass - rm_dim

            # 众数角度
            # 原始: [peakM, ~, peakC] = mode(angBox(:)) [~line 866]
            ang_flat = ang_box.ravel().astype(int)
            ang_flat = ang_flat[ang_flat > 0]
            if len(ang_flat) > 0:
                vals, cnts  = np.unique(ang_flat, return_counts=True)
                best_idx    = int(vals[np.argmax(cnts)]) - 1
                best_idx    = min(max(best_idx, 0), current_global_angle - 1)
                peak_angles = ANGLE_LIST[best_idx].tolist()
            else:
                peak_angles = [0.0, 0.0, 0.0]

            # 评分 / 真实坐标
            # 原始: peakMat(n,10), peakMat(n,1:3) [~line 875–880]
            score       = (max_val - T_mean) / T_std if T_std > 0 else 0.0
            coords_real = sampling_rate * cen_p

            peak_mat.append({
                "score": score,
                "x": float(coords_real[0]), "y": float(coords_real[1]),
                "z": float(coords_real[2]),
                "phi": float(peak_angles[0]), "theta": float(peak_angles[1]),
                "psi": float(peak_angles[2]),
            })

            # 旋转擦除掩膜并清空该区域
            # 原始: rmMask = BH_resample3d(removalMask, peakMat(n,4:6), ...)
            #        mag(...) .* (rmMask < maskCutOff) [~line 885]
            rm_rot   = rotate_volume(removal_mask, *peak_angles)
            i0, i1   = max(i-int(rm_dim[0]), 0), min(i+int(rm_dim[0])+1, size_pick[0])
            j0, j1   = max(j-int(rm_dim[1]), 0), min(j+int(rm_dim[1])+1, size_pick[1])
            k0, k1   = max(k-int(rm_dim[2]), 0), min(k+int(rm_dim[2])+1, size_pick[2])
            rm_sl_g  = cp.asarray(rm_rot[:i1-i0, :j1-j0, :k1-k0])
            mag_g[i0:i1, j0:j1, k0:k1] *= (rm_sl_g < mask_cut_off).astype(cp.float32)

            n_found += 1
            if n_found % 100 == 0:
                print(f"  已找到 {n_found} 个峰值...")
        else:
            # 原始: else → mag(coord) = 0 [~line 900]
            mag_g.ravel()[coord] = 0.0

        max_val = float(cp.max(mag_g))
        coord   = int(cp.argmax(mag_g))

    print(f"[峰值] 共找到 {n_found} 个峰值（尝试 {this_try} 次候选）")
    return peak_mat


# ===========================================================================
# 17. CLI 入口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="emClarity BH_templateSearch3d Python/CuPy GPU reimplement")
    parser.add_argument("--tomo-name", required=True, help="tomogram name (e.g. TS_01)")
    parser.add_argument("--tomo-number", type=int, required=True, help="tomogram number")
    parser.add_argument("--template", required=True, help="template MRC file path")
    parser.add_argument("--wedge-type", type=int, choices=[1, 2, 3, 4], required=True, help="wedge type 1=binary wedge 2=no CTF weight 3=CTF no exposure weight 4=full CTF")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (default 0)")
    parser.add_argument("--whole", action="store_true", help="whole mode: put the entire tomogram into GPU memory at once, the tomogram FFT is calculated only once, faster but requires enough GPU memory. default is chunked mode (memory friendly).")

    args = parser.parse_args()

    template_search_3d(
        tomo_name     = args.tomo_name,
        tomo_number   = args.tomo_number,
        template_path = args.template,
        wedge_type    = args.wedge_type,
        gpu_idx       = args.gpu,
        whole_tomo    = args.whole,
    )


if __name__ == "__main__":
    main()
