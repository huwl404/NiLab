#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fiber2star_v6: Fiber tracking from neural-network prediction volumes.

Input: NN fiber probability MRC (uint8/float32, values 0–255).
  - High values indicate fiber; all fibers share nearly the same direction (≤10°).
  - The NN predicts each Z-slice independently → causes Z-axis discontinuities.

Problems with raw NN output (addressed here):
  1. Threshold=255 → reliable but fragmented
  2. Threshold=254 → continuous but: rough Z-transitions, fibers mis-connected, large blob false-positives

Pipeline
────────────────────────────────────────────────────────────────────────────
  1. Load & normalize        : float32, [0,1]
  2. Z-axis Gaussian smooth  : (GPU) heals slice-to-slice NN discontinuities
  3. Dominant direction      : PCA on high-threshold (near-255) voxels → unit vec
  4. Binary mask             : moderate threshold → ball opening → remove small CCs
  5. Elongation filter       : drop CCs not aligned/elongated along fiber direction
                               (removes blob false-positives and cross-fiber debris)
  7. Skeletonize (CPU)       : 3D thinning → bridge micro-gaps → re-skeletonize
  8. Junction split + filter : break at branch-points → keep branches aligned with
                               dominant direction
  9. Greedy spline extraction: choose best B-spline fiber candidate globally
 10. Spatial erase           : remove all candidate segments within radius-R cylinder
 11. Iterate                 : repeat best-fit + erase until candidates are exhausted
 12. Sample + write .star     : rlnCoordinateX/Y/Z + rlnAngleRot/Tilt/Psi
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import warnings

warnings.filterwarnings(
    action="ignore",
    message=r".*cupyx\.jit\.rawkernel is experimental.*",
    category=FutureWarning,
    module="cupyx.jit._interface",
)

import argparse
from pathlib import Path
from glob import glob, has_magic
import math
import re
import time
import traceback
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import mrcfile
import starfile
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian1d
from cupyx.scipy.ndimage import convolve as cp_convolve
from cucim import skimage as cskimage

from skimage.morphology import skeletonize  # not yet in cucim


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────


def load_mrc(path: Path) -> Tuple[np.ndarray, float]:
    """Load MRC, return (float32 ZYX array, voxel_size_Angstrom)."""
    with mrcfile.open(path, permissive=True, mode="r") as m:
        vol = np.asarray(m.data, dtype=np.float32)
        try:
            vs = float(m.voxel_size.x)
        except Exception:
            vs = 1.0
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    return vol, vs


def save_mrc(path: Path, data: np.ndarray, voxel_size: float = 1.0) -> None:
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(np.asarray(data, dtype=np.float32))
        try:
            m.voxel_size = float(voxel_size)
        except Exception:
            pass


def write_star(out_path: Path, coords_xyz: np.ndarray, angles: np.ndarray, shift_z: float = 0) -> None:
    """Write RELION-style particle STAR (pixel coords, ZYZ Euler degrees)."""
    c = np.asarray(coords_xyz, dtype=np.float32)
    a = np.asarray(angles, dtype=np.float32)
    df = pd.DataFrame({
        "rlnCoordinateX": c[:, 0],
        "rlnCoordinateY": c[:, 1],
        "rlnCoordinateZ": c[:, 2] + shift_z,
        "rlnAngleRot":    a[:, 0],
        "rlnAngleTilt":   a[:, 1],
        "rlnAnglePsi":    a[:, 2],
    })
    starfile.write({"particles": df}, out_path, overwrite=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Z-axis smoothing
# ─────────────────────────────────────────────────────────────────────────────


def smooth_z_axis(vol: np.ndarray, sigma_z: float, sigma_xy: float = 0.0) -> np.ndarray:
    """
    Anisotropic Gaussian smooth (GPU).
    sigma_z > 0 : heal NN slice-to-slice discontinuities along Z.
    sigma_xy > 0: optional mild in-plane smoothing to fill fiber-interior gaps.
    """
    v = cp.asarray(vol, dtype=cp.float32)
    if sigma_z > 0:
        v = cp_gaussian1d(v, sigma=sigma_z, axis=0)
    if sigma_xy > 0:
        v = cp_gaussian1d(v, sigma=sigma_xy, axis=1)
        v = cp_gaussian1d(v, sigma=sigma_xy, axis=2)
    return cp.asnumpy(v).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Dominant fiber direction
# ─────────────────────────────────────────────────────────────────────────────


def _label_sorted_coords(labels_gpu: "cp.ndarray") -> Tuple["cp.ndarray", np.ndarray, int]:
    """
    Shared helper: extract all foreground (label>0) XYZ coordinates, sort by label.

    Returns
    -------
    fg_xyz   : cp.ndarray (N, 3) float64, XYZ coordinates, sorted by label
    bounds   : np.ndarray (n_lab+1,) int64  — bounds[i]:bounds[i+1] = slice for label i+1
    n_lab    : int
    """
    n_lab = int(labels_gpu.max().get())
    fg_zyx = cp.argwhere(labels_gpu > 0)                        # (N, 3)  ZYX
    if len(fg_zyx) == 0:
        return cp.empty((0, 3), dtype=cp.float64), np.array([0]), n_lab

    fg_labs = labels_gpu[fg_zyx[:, 0], fg_zyx[:, 1], fg_zyx[:, 2]]  # (N,) 1-based
    fg_xyz  = fg_zyx[:, [2, 1, 0]].astype(cp.float64)           # (N, 3) XYZ

    # ── Single GPU sort by label (O(N log N) once vs O(n_lab × N) per-label scan) ──
    sort_idx = cp.argsort(fg_labs)
    fg_xyz   = fg_xyz[sort_idx]
    labs_np  = cp.asnumpy(fg_labs[sort_idx]).astype(np.int64)
    bounds   = np.searchsorted(labs_np, np.arange(1, n_lab + 2))  # (n_lab+1,)
    return fg_xyz, bounds, n_lab


def _gpu_cov3(pts_gpu: "cp.ndarray", max_pts: int) -> "cp.ndarray":
    """
    Compute 3×3 covariance matrix for a (n, 3) GPU point array.
    Subsamples uniformly to max_pts when n > max_pts (avoids O(n×9) bottleneck
    for very large CCs while preserving principal direction accuracy).
    """
    n = len(pts_gpu)
    if n > max_pts:
        idx = cp.random.choice(n, max_pts, replace=False)
        pts_gpu = pts_gpu[idx]
    centered = pts_gpu - pts_gpu.mean(axis=0)
    return (centered.T @ centered) / max(len(pts_gpu) - 1, 1)   # (3, 3)


def compute_dominant_direction(
    mask_zyx: np.ndarray,
    min_pts_per_cc: int = 50,
    max_pts_per_cc: int = 50_000,
) -> Optional[np.ndarray]:
    """
    GPU-accelerated dominant fiber direction estimation.

    Approach (inspired by RAPIDS covariance-based PCA):
      1. Single GPU argsort by label  → O(N log N)  [vs O(n_lab × N) per-label argwhere]
      2. 3×3 covariance matrix per CC → O(n_cc × 9)  [vs full N×3 SVD]
      3. Uniform subsampling for large CCs            [max_pts_per_cc]
      4. Batched cp.linalg.eigh on all CCs in one GPU call

    Returns unit vector (x, y, z) or None.
    """
    mask_gpu   = cp.asarray(mask_zyx.astype(bool))
    labels_gpu = cskimage.measure.label(mask_gpu.astype(cp.uint8), connectivity=2)
    fg_xyz, bounds, n_lab = _label_sorted_coords(labels_gpu)

    if len(fg_xyz) == 0:
        return None

    # ── Collect per-CC 3×3 covariance matrices ────────────────────────────────
    cov_list:   List["cp.ndarray"] = []
    length_list: List[float]       = []
    d_gpu = cp.asarray(mask_zyx if n_lab == 0 else np.ones(3))   # placeholder

    for lab_idx in range(n_lab):
        s, e = int(bounds[lab_idx]), int(bounds[lab_idx + 1])
        if e - s < min_pts_per_cc:
            continue
        pts = fg_xyz[s:e]                               # GPU slice — O(CC_size)
        cov_list.append(_gpu_cov3(pts, max_pts_per_cc))

        # Exact length along dominant axis: compute after eigh, use CC count as proxy
        length_list.append(float(e - s))

    if not cov_list:
        # Global fallback — PCA on all foreground
        cov = _gpu_cov3(fg_xyz, max_pts_per_cc)
        try:
            _, evecs = cp.linalg.eigh(cov[None])        # (1, 3, 3)
            return cp.asnumpy(evecs[0, :, 2]).astype(np.float32)
        except Exception:
            return None

    # ── Batched 3×3 eigh — one GPU call for all CCs ───────────────────────────
    cov_batch  = cp.stack(cov_list, axis=0)             # (K, 3, 3)
    evals, evecs = cp.linalg.eigh(cov_batch)            # evals (K,3) ascending
    dirs_gpu   = evecs[:, :, 2]                         # (K, 3) principal directions

    # Exact length proxy: λmax × n  ∝ projected spread²
    counts_gpu = cp.array(length_list, dtype=cp.float64)
    lengths    = cp.asnumpy(cp.sqrt(cp.abs(evals[:, 2]) * counts_gpu))
    dirs_np    = cp.asnumpy(dirs_gpu).astype(np.float64)

    # ── Length-weighted average, sign-consistent ──────────────────────────────
    ref     = dirs_np[int(np.argmax(lengths))]
    signs   = np.where((dirs_np @ ref) >= 0, 1.0, -1.0)
    dom     = (lengths[:, None] * dirs_np * signs[:, None]).sum(0)
    norm    = np.linalg.norm(dom)
    return (dom / norm).astype(np.float32) if norm > 1e-12 else ref.astype(np.float32)


def _draw_direction_volume(
    shape_zyx: Tuple[int, ...],
    dominant_dir_xyz: np.ndarray,
    ball_radius: int = 8,
    line_length: int = 80,
) -> np.ndarray:
    """
    Draw a direction indicator into an empty volume:
      - Large filled ball (value 2) at volume center = origin
      - Line (value 1) from center along dominant_dir_xyz
    Useful for verifying the estimated fiber direction in a MRC viewer.
    """
    vol = np.zeros(shape_zyx, dtype=np.float32)
    cz = shape_zyx[0] // 2
    cy = shape_zyx[1] // 2
    cx = shape_zyx[2] // 2

    # Ball at center
    r = int(ball_radius)
    zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    in_ball = (zz ** 2 + yy ** 2 + xx ** 2) <= r ** 2
    bz = np.clip(cz + zz[in_ball], 0, shape_zyx[0] - 1)
    by = np.clip(cy + yy[in_ball], 0, shape_zyx[1] - 1)
    bx = np.clip(cx + xx[in_ball], 0, shape_zyx[2] - 1)
    vol[bz, by, bx] = 2.0

    # Line from center in dominant direction
    d = np.asarray(dominant_dir_xyz, dtype=np.float64)
    d = d / np.linalg.norm(d)
    for t in np.linspace(0, float(line_length), line_length * 4 + 1):
        lx = int(round(cx + t * d[0]))
        ly = int(round(cy + t * d[1]))
        lz = int(round(cz + t * d[2]))
        if 0 <= lz < shape_zyx[0] and 0 <= ly < shape_zyx[1] and 0 <= lx < shape_zyx[2]:
            if vol[lz, ly, lx] < 1.5:  # don't overwrite ball
                vol[lz, ly, lx] = 1.0

    return vol

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Binary mask extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_binary_mask(
    vol: np.ndarray,
    threshold: float,
    opening_radius: int,
    voxel_size: float = 1.0,
    debug_prefix: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Threshold → ball opening. vol and threshold are in the same raw scale (0–255)."""
    v = cp.asarray(vol, dtype=cp.float32)
    mask = v >= threshold
    n_on = int(mask.sum().get())
    if n_on == 0:
        print("  mask empty after threshold")
        return None
    print(f"  {n_on} voxels ({100.*n_on/v.size:.2f}%)")

    if opening_radius > 0:
        mask = cskimage.morphology.binary_opening(mask, cskimage.morphology.ball(opening_radius))
        if not bool(mask.any().get()):
            print("  mask empty after ball opening")
            return None
        if debug_prefix is not None:
            save_mrc(debug_prefix.parent / (debug_prefix.name + "_3_opened.mrc"),cp.asnumpy(mask.astype(cp.float32)), voxel_size)

    return cp.asnumpy(mask).astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Elongation filter — removes non-fibrous CCs (blob false-positives)
# ─────────────────────────────────────────────────────────────────────────────


def filter_ccs_by_elongation(
    mask_zyx: np.ndarray,
    dominant_dir: np.ndarray,
    min_aspect_ratio: float,
    max_angle_deg: float,
    max_pts_per_cc: int = 50_000,
    voxel_size: float = 1.0,
    debug_prefix: Optional[Path] = None,
) -> np.ndarray:
    """
    GPU-accelerated elongation filter.  Keep CCs that are:
      - PCA axis aligned with dominant_dir  (< max_angle_deg)
      - Aspect ratio (length / perp_span)   > min_aspect_ratio

    Same sort-once optimisation as compute_dominant_direction:
      • Single GPU argsort by label  → no O(n_lab × N) per-label argwhere
      • 3×3 covariance + batched eigh for direction check
      • Per-CC GPU slice for exact length/perp_span (O(CC_size) each)
      • Subsampling for large CCs
    """
    d_gpu   = cp.asarray(dominant_dir, dtype=cp.float64)
    d_np    = cp.asnumpy(d_gpu)
    cos_thr = math.cos(math.radians(max_angle_deg))

    mask_gpu   = cp.asarray(mask_zyx.astype(bool))
    labels_gpu = cskimage.measure.label(mask_gpu.astype(cp.uint8), connectivity=2)
    n_lab      = int(labels_gpu.max().get())
    if n_lab == 0:
        return mask_zyx

    fg_xyz, bounds, _ = _label_sorted_coords(labels_gpu)
    if len(fg_xyz) == 0:
        return mask_zyx

    # ── Pass 1: batched eigh for direction alignment (cheap, GPU) ─────────────
    valid_labs:  List[int]          = []   # 1-based label indices passing size check
    cov_list:    List["cp.ndarray"] = []

    for lab_idx in range(n_lab):
        s, e = int(bounds[lab_idx]), int(bounds[lab_idx + 1])
        if e - s < 10:
            continue
        valid_labs.append(lab_idx + 1)
        cov_list.append(_gpu_cov3(fg_xyz[s:e], max_pts_per_cc))

    if not valid_labs:
        print("  elongation filter: no CCs large enough")
        return np.zeros_like(mask_zyx)

    cov_batch        = cp.stack(cov_list, axis=0)        # (K, 3, 3)
    _, evecs_batch   = cp.linalg.eigh(cov_batch)         # evecs (K, 3, 3)
    main_dirs        = cp.asnumpy(evecs_batch[:, :, 2])  # (K, 3) principal dirs

    # Alignment check (vectorised on CPU — K is small)
    dots    = np.abs(main_dirs @ d_np)                   # (K,)
    aligned = dots >= cos_thr

    # ── Pass 2: exact length + perp_span for aligned CCs (GPU slices) ─────────
    keep          = np.zeros(n_lab + 1, dtype=bool)
    reject_dir    = np.zeros(n_lab + 1, dtype=bool)   # failed direction check
    reject_aspect = np.zeros(n_lab + 1, dtype=bool)   # failed aspect-ratio check
    n_kept = n_rm = 0

    for k, lab in enumerate(valid_labs):
        if not aligned[k]:
            reject_dir[lab] = True
            n_rm += 1
            continue
        s, e   = int(bounds[lab - 1]), int(bounds[lab])
        pts    = fg_xyz[s:e]                             # GPU slice
        if len(pts) > max_pts_per_cc:
            idx = cp.random.choice(len(pts), max_pts_per_cc, replace=False)
            pts = pts[idx]
        centered = pts - pts.mean(axis=0)                # GPU

        # Exact projected length along dominant_dir
        proj      = centered @ d_gpu                     # (n,) GPU
        length    = float((proj.max() - proj.min()).get())
        # Perp span: max distance from the fiber axis line
        r_perp    = centered - proj[:, None] * d_gpu[None, :]   # (n, 3) GPU
        perp_span = float(cp.linalg.norm(r_perp, axis=1).max().get()) * 2 + 1e-3
        if length / perp_span < min_aspect_ratio:
            reject_aspect[lab] = True
            n_rm += 1
            continue

        keep[lab] = True
        n_kept += 1

    n_dir_fail = int((~aligned).sum())
    print(f"  {len(valid_labs)} CCs checked → kept {n_kept} | removed: "
          f"{n_dir_fail} direction, {n_rm - n_dir_fail} aspect")

    labels_np = cp.asnumpy(labels_gpu)
    result    = keep[labels_np].astype(bool)

    if debug_prefix is not None:
        save_mrc(debug_prefix.parent / (debug_prefix.name + "_4_elongated.mrc"), result.astype(np.float32), voxel_size)
        # Rejected CCs: use label value as intensity for easy color-coding in viewer
        rej_dir_vol    = np.where(reject_dir[labels_np],    labels_np.astype(np.float32), 0.0)
        rej_aspect_vol = np.where(reject_aspect[labels_np], labels_np.astype(np.float32), 0.0)
        save_mrc(debug_prefix.parent / (debug_prefix.name + "_4_rejected_direction.mrc"), rej_dir_vol, voxel_size)
        save_mrc(debug_prefix.parent / (debug_prefix.name + "_4_rejected_aspect.mrc"), rej_aspect_vol, voxel_size)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Steps 6: Skeletonize → junction split → direction filter → segments
# ─────────────────────────────────────────────────────────────────────────────


def skeleton_to_segments(
    mask_zyx: np.ndarray,
    dominant_dir: Optional[np.ndarray],
    max_angle_deg: float = 30.0,
    bridge_radius: int = 3,
    junction_clearance: int = 2,
    min_branch_pts: int = 5,
    voxel_size: float = 1.0,
    debug_prefix: Optional[Path] = None,
) -> List[np.ndarray]:
    """
    3D skeletonize → bridge micro-gaps → break at junction voxels →
    direction-filter sub-branches → PCA-ordered segment point arrays.

    Returns list of (N,3) float32 (x,y,z) pixel coord arrays.
    """
    # ---- Skeletonize ----
    skel = skeletonize(mask_zyx.astype(bool))
    if not skel.any():
        print("  skeleton empty")
        return []
    if debug_prefix is not None:
        save_mrc(debug_prefix.parent / (debug_prefix.name + "_5_skel_raw.mrc"), skel.astype(np.float32), voxel_size)

    # ---- Bridge micro-gaps → re-skeletonize ----
    if bridge_radius > 0:
        skel_gpu = cp.asarray(skel)
        dilated = cskimage.morphology.binary_dilation(skel_gpu, cskimage.morphology.ball(bridge_radius))
        skel = skeletonize(cp.asnumpy(dilated).astype(bool))
        if not skel.any():
            return []
        if debug_prefix is not None:
            save_mrc(debug_prefix.parent / (debug_prefix.name + "_6_skel_bridged.mrc"), skel.astype(np.float32), voxel_size)

    skel_gpu = cp.asarray(skel)

    # ---- Junction split + direction filter ----
    if dominant_dir is not None and max_angle_deg < 90.0:
        kernel = cp.ones((3, 3, 3), dtype=cp.int32)
        kernel[1, 1, 1] = 0
        n_nbr = cp_convolve(skel_gpu.astype(cp.int32), kernel, mode="constant", cval=0)
        junc_gpu = skel_gpu & (n_nbr >= 3)
        n_junc = int(junc_gpu.sum().get())

        if n_junc > 0:
            if junction_clearance > 0:
                junc_gpu = cskimage.morphology.binary_dilation(junc_gpu, cskimage.morphology.ball(junction_clearance))
            branches_gpu = skel_gpu & (~junc_gpu)
        else:
            branches_gpu = skel_gpu

        labels_gpu = cskimage.measure.label(branches_gpu.astype(cp.uint8), connectivity=2)
        n_lab = int(labels_gpu.max().get())
        if n_lab == 0:
            return []
        labels_np = cp.asnumpy(labels_gpu)

        all_coords = np.argwhere(labels_np > 0)
        all_labs = labels_np[all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]]
        sort_idx = np.argsort(all_labs)
        sorted_coords = all_coords[sort_idx]
        sorted_labs = all_labs[sort_idx]
        bounds = np.searchsorted(sorted_labs, np.arange(1, n_lab + 2))

        cos_thr = math.cos(math.radians(max_angle_deg))
        keep_mask = np.zeros(n_lab + 1, dtype=bool)
        n_kept = n_rm = n_small = 0

        for lab in range(1, n_lab + 1):
            s, e = bounds[lab - 1], bounds[lab]
            if e - s < min_branch_pts:
                n_small += 1
                continue
            pts = sorted_coords[s:e][:, [2, 1, 0]].astype(np.float32)
            centered = pts - pts.mean(0)
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                bdir = vh[0]
            except Exception:
                continue
            if abs(float(np.dot(bdir, dominant_dir))) >= cos_thr:
                keep_mask[lab] = True
                n_kept += 1
            else:
                n_rm += 1

        print(f"  branches: {n_lab} → kept {n_kept} | removed {n_rm}, small {n_small}")
        filtered = keep_mask[labels_np].astype(bool)
        if not filtered.any():
            return []
        if debug_prefix is not None:
            save_mrc(debug_prefix.parent / (debug_prefix.name + "_7_branches.mrc"), filtered.astype(np.float32), voxel_size)
    else:
        filtered = skel.astype(bool)

    # ---- Regroup via dilation, extract thin-skeleton voxels per group ----
    thin = filtered.astype(bool)
    filt_gpu = cp.asarray(thin)
    if bridge_radius > 0:
        grouped = cskimage.morphology.binary_dilation(filt_gpu, cskimage.morphology.ball(bridge_radius))
    else:
        grouped = filt_gpu

    final_labels_gpu = cskimage.measure.label(grouped.astype(cp.uint8), connectivity=2)
    n_final = int(final_labels_gpu.max().get())
    if n_final == 0:
        return []

    final_labels_np = cp.asnumpy(final_labels_gpu)
    thin_labels = final_labels_np.copy()
    thin_labels[~thin] = 0

    all_coords = np.argwhere(thin_labels > 0)
    all_labs = thin_labels[all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]]
    sort_idx = np.argsort(all_labs)
    sorted_coords = all_coords[sort_idx]
    sorted_labs = all_labs[sort_idx]
    bounds = np.searchsorted(sorted_labs, np.arange(1, n_final + 2))

    segments: List[np.ndarray] = []
    for lab in range(1, n_final + 1):
        s, e = bounds[lab - 1], bounds[lab]
        if e - s < 2:
            continue
        pts_xyz = sorted_coords[s:e][:, [2, 1, 0]].astype(np.float32)
        centered = pts_xyz - pts_xyz.mean(0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ vh[0]
        except Exception:
            continue
        pts_sorted = pts_xyz[np.argsort(proj)]
        # Deduplicate spatially
        dedup = [0]
        for i in range(1, len(pts_sorted)):
            if np.linalg.norm(pts_sorted[i] - pts_sorted[dedup[-1]]) >= 1.0:
                dedup.append(i)
        if dedup[-1] != len(pts_sorted) - 1:
            dedup.append(len(pts_sorted) - 1)
        pts = pts_sorted[dedup]
        if len(pts) >= 2:
            segments.append(pts.astype(np.float32))

    print(f"  {n_final} groups → {len(segments)} segments")
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Step 7b: Greedy B-spline extraction with spatial erase
# ─────────────────────────────────────────────────────────────────────────────


def _safe_unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    nv = float(np.linalg.norm(v))
    if not np.isfinite(nv) or nv < 1e-8:
        return fallback.copy()
    return (v / nv).astype(np.float64)


def _polyline_arc_length(pts_xyz: np.ndarray) -> float:
    if len(pts_xyz) < 2:
        return 0.0
    d = np.diff(pts_xyz.astype(np.float64), axis=0)  # np.diff 得到相邻段向量。
    return float(np.sum(np.linalg.norm(d, axis=1)))  # 每段求欧氏长度并求和 = 折线总弧长。


def _segment_dir(seg_xyz: np.ndarray, dominant_dir: np.ndarray) -> np.ndarray:
    seg = seg_xyz.astype(np.float64)
    if len(seg) >= 3:
        try:
            _, _, vh = np.linalg.svd(seg - seg.mean(0), full_matrices=False)
            d = _safe_unit(vh[0], dominant_dir)  # 优先用 SVD 的第一主轴 vh[0] 作为方向。
        except Exception:
            d = _safe_unit(seg[-1] - seg[0], dominant_dir)
    else:
        d = _safe_unit(seg[-1] - seg[0], dominant_dir)
    if float(np.dot(d, dominant_dir)) < 0.0:
        d = -d  # 确保方向与 dominant_dir 同向。
    return d


def _orient_segment_along(seg_xyz: np.ndarray, dominant_dir: np.ndarray) -> np.ndarray:
    seg = seg_xyz.astype(np.float64)
    if len(seg) < 2:
        return seg
    p = seg @ dominant_dir  # 计算 segment 起点到终点沿 dominant_dir 方向的投影长度。
    if p[-1] < p[0]:
        seg = seg[::-1]  # 如果终点在前，起点在后，则反转 segment。
    return seg


def _stitch_chain_points(chain: List[int], segments: List[np.ndarray], dominant_dir: np.ndarray) -> np.ndarray:
    if not chain:
        return np.empty((0, 3), dtype=np.float64)
    pts_list: List[np.ndarray] = []
    for idx in chain:
        pts_list.append(_orient_segment_along(segments[idx], dominant_dir))  # 将每个 segment 沿 dominant_dir 方向排列。
    pts = np.vstack(pts_list)
    pts = pts[np.argsort(pts @ dominant_dir)]  # 按投影长度排序。
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= 0.7:  # 如果当前点与上一个保留点距离 >= 0.7，则保留。
            keep.append(i)
    if keep[-1] != len(pts) - 1:
        keep.append(len(pts) - 1)
    return pts[keep].astype(np.float64)


def _sample_volume_nn(vol_zyx: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    """Sample volume values at given points using nearest neighbor interpolation."""
    if len(pts_xyz) == 0:
        return np.empty((0,), dtype=np.float64)
    z = np.clip(np.round(pts_xyz[:, 2]).astype(np.int64), 0, vol_zyx.shape[0] - 1)
    y = np.clip(np.round(pts_xyz[:, 1]).astype(np.int64), 0, vol_zyx.shape[1] - 1)
    x = np.clip(np.round(pts_xyz[:, 0]).astype(np.int64), 0, vol_zyx.shape[2] - 1)
    return vol_zyx[z, y, x].astype(np.float64)


def _fit_bspline_curve(
    pts_xyz: np.ndarray,
    smoothness: float,
    min_eval: int = 40,
) -> Tuple[np.ndarray, float, float]:
    """Fit a 3D B-spline curve and return: (curve points, bending energy, curvature CV)."""
    pts = np.asarray(pts_xyz, dtype=np.float64)
    if len(pts) < 4:
        return pts.astype(np.float32), 0.0, 0.0
    try:
        k = min(3, len(pts) - 1) # 最大三次样条。
        s = max(float(smoothness), 0.0) * float(len(pts))  # 平滑因子：越大越平滑。
        tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=s, k=k)
        n_eval = max(min_eval, len(pts) * 3)  # 采样点数：至少 min_eval，但不超过点数的三倍。
        u = np.linspace(0.0, 1.0, n_eval)  # 均匀采样弧长参数。
        x, y, z = splev(u, tck)  # 计算 B-spline 曲线。
        curve = np.stack([x, y, z], axis=1).astype(np.float64)

        d1 = np.stack(splev(u, tck, der=1), axis=1)  # 计算一阶导数。
        d2 = np.stack(splev(u, tck, der=2), axis=1)  # 计算二阶导数。
        sp = np.linalg.norm(d1, axis=1)  # 计算一阶导数的模。
        cross = np.cross(d1, d2)  # 计算曲率。
        curv = np.linalg.norm(cross, axis=1) / np.maximum(sp ** 3, 1e-8)  # 计算曲率。

        du = 1.0 / max(n_eval - 1, 1)  # 采样步长。
        ds = sp * du  # 计算弧长。
        total_len = max(float(np.sum(ds)), 1e-8)  # 总弧长。
        bending = float(np.sum((curv ** 2) * ds) / total_len)  # 计算弯曲能量，越大越弯曲。
        curv_mean = float(np.mean(curv))  # 计算曲率均值。
        curv_cv = float(np.std(curv) / max(curv_mean, 1e-6))  # 计算曲率变异系数，越大变化越大。
        return curve.astype(np.float32), bending, curv_cv
    except Exception:
        return pts.astype(np.float32), 0.0, 0.0


def _grow_chain_from_seed(
    seed_idx: int,
    active_idxs: List[int],
    centroids: np.ndarray,
    seg_dirs: np.ndarray,
    seg_signals: np.ndarray,
    dominant_dir: np.ndarray,
    fiber_radius_px: float,
    max_gap_along: float,
    cos_thr: float,
) -> List[int]:
    active_set = set(active_idxs)
    chain = [seed_idx]
    used: Set[int] = {seed_idx}

    def _best_next(cur_idx: int, forward: bool) -> Optional[int]:
        c0 = centroids[cur_idx]
        d0 = seg_dirs[cur_idx]
        best = None
        best_score = -1e18
        for j in active_set:  # 遍历所有活跃段。
            if j in used:
                continue
            dv = centroids[j] - c0  # 计算当前段与目标段的向量。
            gap_along = float(np.dot(dv, dominant_dir))  
            if forward:
                if gap_along <= 1e-6 or gap_along > max_gap_along:
                    continue
            else:
                if gap_along >= -1e-6 or -gap_along > max_gap_along:
                    continue
            lateral = float(np.linalg.norm(dv - gap_along * dominant_dir))  # 计算侧向偏移。
            if lateral > float(fiber_radius_px):
                continue
            gap_norm = float(np.linalg.norm(dv))  # 计算向量模。
            if gap_norm < 1e-6:
                continue
            cos_gap = abs(gap_along) / gap_norm  # 计算连接方向对齐度。
            if cos_gap < cos_thr:
                continue
            if abs(float(np.dot(seg_dirs[j], dominant_dir))) < cos_thr:  # 计算段方向对齐度。
                continue
            turn_align = abs(float(np.dot(seg_dirs[j], d0)))
            score = (
                2.0 * float(seg_signals[j])  # 信号强度权重。
                + 50.0 * turn_align  # 段方向对齐权重。
                + 30.0 * cos_gap  # 连接方向对齐权重。
                - 1.5 * lateral  # 侧向偏移惩罚。
                - 0.4 * abs(gap_along)  # 跨距偏移惩罚。
            )
            if score > best_score:  # 如果当前分数大于最佳分数，则更新最佳分数和最佳索引。
                best_score = score
                best = j  # 更新最佳索引。
        return best

    cur = seed_idx
    while True:
        nxt = _best_next(cur, forward=True)
        if nxt is None:
            break
        chain.append(nxt)
        used.add(nxt)
        cur = nxt

    cur = seed_idx
    head: List[int] = []
    while True:
        prv = _best_next(cur, forward=False)
        if prv is None:
            break
        head.append(prv)
        used.add(prv)
        cur = prv

    head.reverse()
    return head + chain


def _segment_touches_curve(seg_xyz: np.ndarray, curve_xyz: np.ndarray, radius_px: float) -> bool:
    """Check if a segment touches a curve within a given radius."""
    if len(seg_xyz) == 0 or len(curve_xyz) == 0:
        return False
    r2 = float(radius_px) * float(radius_px)
    seg_min = seg_xyz.min(axis=0) - radius_px
    seg_max = seg_xyz.max(axis=0) + radius_px
    cur_min = curve_xyz.min(axis=0)
    cur_max = curve_xyz.max(axis=0)
    if np.any(seg_max < cur_min) or np.any(cur_max < seg_min):
        return False

    curve_sub = curve_xyz[::2] if len(curve_xyz) > 120 else curve_xyz # 如果曲线点数大于120，则每隔一个点取一个点。
    diff = seg_xyz[:, None, :] - curve_sub[None, :, :]
    d2 = np.sum(diff * diff, axis=2)  # 计算每个点到曲线的距离的平方。
    return bool(np.any(d2 <= r2))


def extract_fibers_by_greedy_bspline(
    segments: List[np.ndarray],
    vol_zyx: np.ndarray,
    dominant_dir: np.ndarray,
    fiber_radius_px: float,
    erase_radius_px: float,
    min_len_px: float,
    max_angle_deg: float = 30.0,
    smoothness: float = 0.4,
    signal_weight: float = 2.5,
    bending_weight: float = 0.05,
    curvature_cv_weight: float = 0.2,
    min_candidate_score: float = 0.0,
) -> List[np.ndarray]:
    """
    Iterative greedy extraction:
      1) evaluate best spline candidate from current segment pool
      2) erase all segments inside radius-R cylinder around chosen spline
      3) repeat until no valid candidate remains
    """
    if not segments:
        return []

    d = np.asarray(dominant_dir, dtype=np.float64)
    d /= max(float(np.linalg.norm(d)), 1e-8)
    cos_thr = math.cos(math.radians(max_angle_deg))
    max_gap_along = max(8.0 * float(fiber_radius_px), 12.0)

    centroids = np.array([seg.mean(0) for seg in segments], dtype=np.float64)
    seg_dirs = np.array([_segment_dir(seg, d) for seg in segments], dtype=np.float64)  # 计算每个段的朝向。
    seg_lens = np.array([_polyline_arc_length(seg) for seg in segments], dtype=np.float64)  # 计算每个段的弧长。
    seg_signals = np.array([float(_sample_volume_nn(vol_zyx, seg).mean()) for seg in segments], dtype=np.float64)  # 计算每个段的信号强度。

    active: Set[int] = set(range(len(segments)))
    fibers: List[np.ndarray] = []  # 存储提取的纤维。
    iter_idx = 0

    while active:
        iter_idx += 1
        active_list = sorted(active)
        best_score = -1e18
        best_curve = None
        best_chain: List[int] = []
        best_arc = 0.0

        for seed in active_list:
            chain = _grow_chain_from_seed(
                seed_idx=seed,
                active_idxs=active_list,
                centroids=centroids,
                seg_dirs=seg_dirs,
                seg_signals=seg_signals,
                dominant_dir=d,
                fiber_radius_px=fiber_radius_px,
                max_gap_along=max_gap_along,
                cos_thr=cos_thr,
            )
            if len(chain) == 0:
                continue

            chain_pts = _stitch_chain_points(chain, segments, d)
            if len(chain_pts) < 4:
                continue
            arc = _polyline_arc_length(chain_pts)
            if arc < float(min_len_px):
                continue

            curve, bending, curv_cv = _fit_bspline_curve(chain_pts, smoothness=smoothness)
            if len(curve) < 2:
                continue
            curve_signal = float(_sample_volume_nn(vol_zyx, curve).mean()) if len(curve) else 0.0
            chain_strength = float(np.mean(seg_signals[np.asarray(chain, dtype=np.int64)]))
            chain_len = float(np.sum(seg_lens[np.asarray(chain, dtype=np.int64)]))
            score = (
                signal_weight * (0.55 * curve_signal + 0.45 * chain_strength)
                + 0.05 * chain_len
                + 0.25 * len(chain)
                - bending_weight * bending
                - curvature_cv_weight * curv_cv
            )
            if score > best_score:
                best_score = score
                best_curve = curve.astype(np.float32)
                best_chain = chain
                best_arc = arc

        if best_curve is None or best_score < float(min_candidate_score):
            break

        fibers.append(best_curve)
        erased = 0
        for idx in list(active):
            # 如果当前段在最佳链中或与最佳曲线相交，则删除当前段。
            if idx in best_chain or _segment_touches_curve(segments[idx].astype(np.float64), best_curve.astype(np.float64), erase_radius_px):
                active.remove(idx)
                erased += 1

        print(
            f"  greedy iter {iter_idx:02d}: chain={len(best_chain)} seg, "
            f"arc={best_arc:.1f}px, score={best_score:.3f}, erased={erased}, remain={len(active)}"
        )
    return fibers


# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Sampling + Euler angles
# ─────────────────────────────────────────────────────────────────────────────


def fit_line_and_sample(
    segment_xyz: np.ndarray,
    dominant_dir: np.ndarray,
    spacing_px: float,
    pre_smoothed: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a mild-curvature centerline and resample by arc length.
    Unlike direct polyline interpolation, this does not need to pass each local
    segment point, which helps suppress over-bending from skeleton jitter.
    Returns (coords (N,3) float32 xyz, global_dir (3,) float32 xyz).
    """
    def _unit(v: np.ndarray) -> np.ndarray:
        nv = float(np.linalg.norm(v))
        if nv < 1e-8 or not np.isfinite(nv):
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return v / nv

    def _resample_polyline(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(poly) == 0:
            d0 = _unit(np.asarray(dominant_dir, dtype=np.float64))
            return np.empty((0, 3), dtype=np.float32), d0.astype(np.float32)
        if len(poly) == 1:
            d0 = _unit(np.asarray(dominant_dir, dtype=np.float64))
            return poly.astype(np.float32), d0.astype(np.float32)

        seg_vec = np.diff(poly, axis=0)
        seg_len = np.linalg.norm(seg_vec, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = float(cum[-1])
        if total_len < max(spacing_px * 0.5, 1e-6):
            mid = poly.mean(0, keepdims=True)
            d = _unit(poly[-1] - poly[0])
            if float(np.dot(d, dominant_dir)) < 0:
                d = -d
            return mid.astype(np.float32), d.astype(np.float32)

        n_pts = max(2, int(round(total_len / spacing_px)) + 1)
        ts = np.linspace(0.0, total_len, n_pts)
        out = np.empty((n_pts, 3), dtype=np.float64)
        j = 0
        for k, t in enumerate(ts):
            while j + 1 < len(cum) and cum[j + 1] < t:
                j += 1
            if j + 1 >= len(cum):
                out[k] = poly[-1]
                continue
            den = max(cum[j + 1] - cum[j], 1e-8)
            a = (t - cum[j]) / den
            out[k] = (1.0 - a) * poly[j] + a * poly[j + 1]

        gd = _unit(out[-1] - out[0])
        if float(np.dot(gd, dominant_dir)) < 0:
            gd = -gd
        return out.astype(np.float32), gd.astype(np.float32)

    pts = np.asarray(segment_xyz, dtype=np.float64)
    if len(pts) == 0:
        d0 = _unit(np.asarray(dominant_dir, dtype=np.float64))
        return np.empty((0, 3), dtype=np.float32), d0.astype(np.float32)

    # Deduplicate tiny spatial repeats.
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= 1e-6:
            keep.append(i)
    pts = pts[keep]
    if len(pts) < 4:
        return _resample_polyline(pts)

    # Mild smoothing for noisy raw polylines before model fitting.
    if pre_smoothed:
        pts_smooth = pts
    else:
        win = min(7, len(pts) if len(pts) % 2 == 1 else len(pts) - 1)
        if win >= 3:
            r = win // 2
            sm = np.zeros_like(pts)
            for i in range(len(pts)):
                s = max(0, i - r)
                e = min(len(pts), i + r + 1)
                sm[i] = pts[s:e].mean(0)
            pts_smooth = sm
        else:
            pts_smooth = pts

    # Build local frame: d (dominant), u/v (perpendicular).
    d = _unit(np.asarray(dominant_dir, dtype=np.float64))
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(ref, d))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = _unit(ref - float(np.dot(ref, d)) * d)
    v = _unit(np.cross(d, u))

    origin = pts_smooth.mean(axis=0)
    rel = pts_smooth - origin[None, :]
    t = rel @ d
    tu = rel @ u
    tv = rel @ v
    span = float(np.max(t) - np.min(t))
    if span < 2.0:
        return _resample_polyline(pts_smooth)

    # Bin/aggregate to avoid fitting every local jitter point.
    n_bins = min(24, max(6, len(t) // 8))
    edges = np.linspace(float(np.min(t)), float(np.max(t)), n_bins + 1)
    tb, ub, vb = [], [], []
    for bi in range(n_bins):
        if bi < n_bins - 1:
            m = (t >= edges[bi]) & (t < edges[bi + 1])
        else:
            m = (t >= edges[bi]) & (t <= edges[bi + 1])
        if not np.any(m):
            continue
        tb.append(float(np.median(t[m])))
        ub.append(float(np.median(tu[m])))
        vb.append(float(np.median(tv[m])))

    if len(tb) < 4:
        return _resample_polyline(pts_smooth)

    tb = np.asarray(tb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    vb = np.asarray(vb, dtype=np.float64)

    t0 = float(np.mean(tb))
    t_scale = max(float(np.std(tb)), 1.0)
    ts = (tb - t0) / t_scale
    A = np.stack([np.ones_like(ts), ts, ts * ts], axis=1)

    # Penalize quadratic term to keep only slight curvature.
    lam2 = 1.5 if pre_smoothed else 2.0
    reg = np.diag([1e-6, 1e-4, lam2])

    try:
        # 2-step robust weighted least squares.
        w = np.ones(len(tb), dtype=np.float64)
        coef_u = np.zeros(3, dtype=np.float64)
        coef_v = np.zeros(3, dtype=np.float64)
        for _ in range(2):
            W = np.sqrt(w)[:, None]
            Aw = A * W
            uw = ub * np.sqrt(w)
            vw = vb * np.sqrt(w)
            coef_u = np.linalg.solve(Aw.T @ Aw + reg, Aw.T @ uw)
            coef_v = np.linalg.solve(Aw.T @ Aw + reg, Aw.T @ vw)
            ru = ub - A @ coef_u
            rv = vb - A @ coef_v
            rr = np.sqrt(ru * ru + rv * rv)
            s = max(float(np.median(rr) * 1.4826), 1e-3)
            w = 1.0 / (1.0 + (rr / (2.5 * s)) ** 2)
    except Exception:
        return _resample_polyline(pts_smooth)

    # Evaluate smooth centerline.
    t_min = float(np.percentile(t, 1))
    t_max = float(np.percentile(t, 99))
    n_dense = max(60, int(round((t_max - t_min) * 2.0)))
    t_grid = np.linspace(t_min, t_max, n_dense)
    ts_grid = (t_grid - t0) / t_scale
    G = np.stack([np.ones_like(ts_grid), ts_grid, ts_grid * ts_grid], axis=1)
    gu = G @ coef_u
    gv = G @ coef_v
    curve = origin[None, :] + t_grid[:, None] * d[None, :] + gu[:, None] * u[None, :] + gv[:, None] * v[None, :]

    return _resample_polyline(curve.astype(np.float64))


def vector_to_euler_zyz(vec_xyz: np.ndarray) -> Tuple[float, float, float]:
    """Direction vector → RELION ZYZ Euler angles (rot, tilt, psi) in degrees."""
    v = np.asarray(vec_xyz, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-6:
        return 0.0, 0.0, 0.0
    v /= n
    z_ref = np.array([0., 0., 1.], dtype=np.float64)
    try:
        rot_obj, _ = R.align_vectors(v[None, :], z_ref[None, :])
    except Exception:
        axis = np.cross(z_ref, v)
        s = float(np.linalg.norm(axis))
        c = float(np.clip(np.dot(z_ref, v), -1., 1.))
        if s < 1e-8:
            rot_obj = R.identity() if c > 0 else R.from_rotvec(np.pi * np.array([1., 0., 0.]))
        else:
            axis /= s
            kx, ky, kz = axis
            K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
            rot_obj = R.from_matrix(np.eye(3) + K * s + (K @ K) * (1 - c))
    ang = rot_obj.as_euler("ZYZ", degrees=True)
    return float(ang[0]), float(ang[1]), float(ang[2])


def _draw_segments_to_volume(
    segments: List[np.ndarray],
    shape_zyx: Tuple[int, ...],
    radius: int = 0,
    bridge_gaps: bool = False,
) -> np.ndarray:
    """
    Color-code segments into a float32 label volume for visualization.

    radius=0       → 1-voxel-wide lines
    radius>0       → filled sphere of that radius at every point (use 2 for 4px)
    bridge_gaps    → also interpolate a 1-voxel line between consecutive points
                     that are more than 1.5 px apart (= merged gap bridges).
                     Use this for the post-merge visualization so it looks
                     different from the pre-merge one (bridges fill the gaps).
    """
    vol = np.zeros(shape_zyx, dtype=np.float32)

    def _draw_point(cx: int, cy: int, cz: int, val: float) -> None:
        if radius <= 0:
            if 0 <= cz < shape_zyx[0] and 0 <= cy < shape_zyx[1] and 0 <= cx < shape_zyx[2]:
                vol[cz, cy, cx] = val
        else:
            r = int(radius)
            offsets = _draw_point._offsets  # type: ignore[attr-defined]
            zs = np.clip(cz + offsets[:, 0], 0, shape_zyx[0] - 1)
            ys = np.clip(cy + offsets[:, 1], 0, shape_zyx[1] - 1)
            xs = np.clip(cx + offsets[:, 2], 0, shape_zyx[2] - 1)
            vol[zs, ys, xs] = val

    if radius > 0:
        r = int(radius)
        _draw_point._offsets = (  # type: ignore[attr-defined]
            np.argwhere(sum((np.mgrid[-r:r+1, -r:r+1, -r:r+1][i] ** 2 for i in range(3))) <= r * r) - r)

    for idx, seg in enumerate(segments, 1):
        val = float(idx)
        for pt in seg:
            _draw_point(int(round(float(pt[0]))), int(round(float(pt[1]))), int(round(float(pt[2]))), val)

        if bridge_gaps and len(seg) >= 2:
            for k in range(len(seg) - 1):
                p1, p2 = seg[k].astype(np.float64), seg[k + 1].astype(np.float64)
                gap = float(np.linalg.norm(p2 - p1))
                if gap > 1.5:                              # only draw bridges (not normal adjacent pts)
                    n_steps = max(int(gap), 2)
                    for t in np.linspace(0.0, 1.0, n_steps + 1)[1:-1]:
                        pt_b = p1 + t * (p2 - p1)
                        _draw_point(int(round(pt_b[0])), int(round(pt_b[1])),
                                    int(round(pt_b[2])), val)

    return vol


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline per tomogram
# ─────────────────────────────────────────────────────────────────────────────


def _process_single_tomogram(args: argparse.Namespace, tomo_path: Path, out_prefix: Path) -> None:
    t0 = time.time()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f"loading: {tomo_path}")
    vol, mrc_voxel = load_mrc(tomo_path)
    voxel = float(args.voxel_size) if float(args.voxel_size) > 0 else mrc_voxel
    print(f"  shape (z,y,x): {vol.shape}, voxel: {voxel:.3f} Å, value range: [{vol.min():.0f}, {vol.max():.0f}]")

    # ── 2. Z-axis smoothing ───────────────────────────────────────────────────
    if args.sigma_z > 0 or args.sigma_xy > 0:
        print(f"smoothing: sigma_z={args.sigma_z}, sigma_xy={args.sigma_xy}")
        vol = smooth_z_axis(vol, sigma_z=args.sigma_z, sigma_xy=args.sigma_xy)
        if args.debug:
            save_mrc(out_prefix.parent / (out_prefix.name + "_1_smoothed.mrc"), vol, voxel)

    # ── 3. Dominant direction from high-confidence voxels ─────────────────────
    high_th = float(args.high_threshold)
    print(f"estimating dominant direction: high_threshold={high_th}")
    high_mask = vol >= high_th
    n_high = int(high_mask.sum())
    dominant_dir = None
    if n_high >= 10:
        dominant_dir = compute_dominant_direction(high_mask, min_pts_per_cc=20)
    
    if dominant_dir is not None:
        print(f"  dominant direction (xyz): [{dominant_dir[0]:.3f}, {dominant_dir[1]:.3f}, {dominant_dir[2]:.3f}]")
    else:
        print("no dominant direction found; exiting")
        return
    
    if args.debug:
        save_mrc(out_prefix.parent / (out_prefix.name + "_2_high_mask.mrc"), high_mask.astype(np.float32), voxel)
        dir_vol = _draw_direction_volume(vol.shape, dominant_dir)
        save_mrc(out_prefix.parent / (out_prefix.name + "_2_dom_direction.mrc"), dir_vol, voxel)

    # ── 4. Binary mask ────────────────────────────────────────────────────────
    main_th = float(args.threshold)
    print(f"binary mask: threshold={main_th}, opening_radius={args.opening_radius}")
    mask = extract_binary_mask(
        vol, main_th,
        opening_radius=args.opening_radius,
        voxel_size=voxel,
        debug_prefix=out_prefix if args.debug else None,
    )
    if mask is None:
        print("no mask extracted; exiting")
        return
    vol_shape = mask.shape

    # ── 5. Elongation filter ─────────────────────────────────────────────────
    print(f"elongation filter: min_aspect={args.min_aspect_ratio}, max_angle={args.direction_filter_angle}°")
    mask = filter_ccs_by_elongation(
        mask, dominant_dir,
        min_aspect_ratio=args.min_aspect_ratio,
        max_angle_deg=args.direction_filter_angle,
        voxel_size=voxel,
        debug_prefix=out_prefix if args.debug else None,
    )
    if not mask.any():
        print("  all CCs removed by elongation filter; exiting")
        return

    # ── 6. Skeletonize → segments ───────────────────────────────────────────
    print(f"skeletonizing: bridge_radius={args.bridge_radius}, direction_filter={args.direction_filter_angle}°")
    segments = skeleton_to_segments(
        mask, dominant_dir,
        max_angle_deg=args.direction_filter_angle,
        bridge_radius=args.bridge_radius,
        junction_clearance=2,
        min_branch_pts=5,
        voxel_size=voxel,
        debug_prefix=out_prefix if args.debug else None,
    )
    if not segments:
        print("no segments found; exiting")
        return
    if args.debug:
        save_mrc(out_prefix.parent / (out_prefix.name + "_8_segments.mrc"), _draw_segments_to_volume(segments, vol_shape, radius=2), voxel)

    # ── 7. Greedy B-spline extraction + spatial erase ────────────────────────
    fiber_radius_px = float(args.fiber_diameter) / 2.0 / voxel
    erase_radius_px = (
        float(args.erase_radius) / voxel
        if float(args.erase_radius) > 0
        else max(fiber_radius_px * float(args.erase_radius_scale), fiber_radius_px)
    )
    min_len_px = max(0.0, float(args.min_fiber_length) / voxel)
    print(
        "greedy spline extraction: "
        f"fiber_radius={fiber_radius_px:.1f}px, erase_radius={erase_radius_px:.1f}px, "
        f"angle<={args.direction_filter_angle} deg, min_len={min_len_px:.1f}px"
    )
    fibers = extract_fibers_by_greedy_bspline(
        segments,
        vol_zyx=vol,
        dominant_dir=dominant_dir,
        fiber_radius_px=fiber_radius_px,
        erase_radius_px=erase_radius_px,
        min_len_px=min_len_px,
        max_angle_deg=args.direction_filter_angle,
        smoothness=args.spline_smoothness,
        signal_weight=args.signal_weight,
        bending_weight=args.bending_weight,
        curvature_cv_weight=args.curvature_consistency_weight,
        min_candidate_score=args.min_candidate_score,
    )
    print(f"  {len(segments)} segments → {len(fibers)} extracted fibers")
    if args.debug:
        save_mrc(out_prefix.parent / (out_prefix.name + "_9_extracted_fibers.mrc"), _draw_segments_to_volume(fibers, vol_shape, radius=2, bridge_gaps=True), voxel)

    if not fibers:
        print("no fibers remain; exiting")
        return

    # ── 8. Sample + write STAR ───────────────────────────────────────────────
    spacing_px = float(args.spacing) / voxel
    all_coords: List[np.ndarray] = []
    all_angles: List[np.ndarray] = []
    fallback_dir = dominant_dir if dominant_dir is not None else np.array([1., 0., 0.], np.float32)

    for fiber in fibers:
        pts, line_dir = fit_line_and_sample(fiber, fallback_dir, spacing_px, pre_smoothed=True)
        if len(pts) == 0:
            continue
        rot, tilt, psi = vector_to_euler_zyz(line_dir)
        angs = np.tile([rot, tilt, psi], (len(pts), 1)).astype(np.float32)
        all_coords.append(pts)
        all_angles.append(angs)

    if not all_coords:
        print("no particles sampled; STAR not written")
        return

    coords = np.vstack(all_coords).astype(np.float32)
    angles = np.vstack(all_angles).astype(np.float32)
    star_path = out_prefix.parent / (out_prefix.name + "_particles.star")
    write_star(star_path, coords, angles, shift_z=args.shift_z)
    print(f"saved {len(coords)} particles → {star_path}  ({time.time()-t0:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution (shared with v4)
# ─────────────────────────────────────────────────────────────────────────────


def _looks_like_regex(s: str) -> bool:
    return bool(re.search(r"(\\.|\\d|\\D|\\w|\\W|\\s|\\S|[\^\$\+\?\|\(\)\[\]\{\}])", s))


def _resolve_paths(tomo_arg: str, recursive: bool) -> List[Path]:
    raw = Path(tomo_arg).expanduser()
    if not recursive:
        p = (raw if raw.is_absolute() else Path.cwd() / raw).resolve()
        if not p.exists():
            raise SystemExit(f"error: not found: {p}")
        return [p]

    if has_magic(tomo_arg):
        pattern = str(raw if raw.is_absolute() else Path.cwd() / raw)
        matched = [Path(p).resolve() for p in glob(pattern)]
    elif _looks_like_regex(tomo_arg):
        parent = (raw.parent if raw.is_absolute() else (Path.cwd() / raw.parent)).resolve()
        try:
            name_re = re.compile(raw.name)
        except re.error as e:
            raise SystemExit(f"error: bad regex: {e}")
        matched = [p.resolve() for p in parent.rglob("*.mrc") if name_re.fullmatch(p.name)]
    else:
        target = (raw if raw.is_absolute() else Path.cwd() / raw).resolve()
        matched = list(target.rglob("*.mrc")) if target.is_dir() else ([target] if target.exists() else [])

    paths = sorted(matched)
    if not paths:
        raise SystemExit(f"error: no files matched for --tomo={tomo_arg}")
    return paths


def _resolve_out_prefix(args: argparse.Namespace, tomo_path: Path, multi: bool) -> Path:
    if args.out_prefix is not None:
        p = Path(args.out_prefix)
        if not p.is_absolute():
            p = tomo_path.parent / p
        if multi:
            p = p.with_name(f"{p.name}_{tomo_path.stem}")
        return p
    return tomo_path.parent / tomo_path.stem


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="fiber2star_v6: fiber tracking from AIS prediction volumes to particles .star")
    p.add_argument("--tomo", required=True, help="input AIS prediction .mrc (0-255); supports glob/regex with --recursive")
    p.add_argument("--voxel-size", type=float, default=-1.0, help="voxel size in Å (default: read from MRC header)")
    p.add_argument("--sigma-z", type=float, default=1.5, help="Gaussian sigma along Z to bridge slice discontinuities (default 1.5)")
    p.add_argument("--sigma-xy", type=float, default=0.0, help="optional mild in-plane Gaussian sigma (default 0 = disabled)")
    p.add_argument("--high-threshold", type=float, default=254.0, help="threshold for dominant direction estimation (default 254)")
    p.add_argument("--threshold", type=float, default=252.0, help="threshold for main binary mask (default 252)")
    p.add_argument("--opening-radius", type=int, default=3, help="ball opening radius to remove thin bridges (default 3)")
    p.add_argument("--min-aspect-ratio", type=float, default=2.0, help="minimum length/cross-span aspect ratio for elongation filter (default 2)")
    p.add_argument("--direction-filter-angle", type=float, default=20.0, help="max angle from dominant fiber direction for branch/CC filtering (default 20°)")
    p.add_argument("--bridge-radius", type=int, default=3, help="dilation radius to bridge skeleton micro-gaps (default 3)")
    p.add_argument("--fiber-diameter", type=float, default=500.0, help="fiber diameter in Å used for local neighborhood radius (default 500)")
    p.add_argument("--erase-radius", type=float, default=-1.0, help="absolute erase radius in Å (default: auto from --erase-radius-scale)")
    p.add_argument("--erase-radius-scale", type=float, default=1.1, help="erase radius scale relative to fiber radius when --erase-radius<=0")
    p.add_argument("--min-fiber-length", type=float, default=1000.0, help="minimum arc length in Å for each greedy spline candidate (default 1000)")
    p.add_argument("--spline-smoothness", type=float, default=1.5, help="B-spline smoothing strength (larger means smoother, default 1.5)")
    p.add_argument("--signal-weight", type=float, default=15.0, help="weight of signal-confidence term in greedy score (default 15.0)")
    p.add_argument("--bending-weight", type=float, default=0.05, help="weight of B-spline bending-energy penalty (default 0.05)")
    p.add_argument("--curvature-consistency-weight", type=float, default=0.2, help="weight of curvature-consistency penalty (default 0.2)")
    p.add_argument("--min-candidate-score", type=float, default=0.0, help="stop extraction when best candidate score drops below this value")
    p.add_argument("--spacing", type=float, default=40, help="particle sampling spacing along fiber (Å, default 40)")
    p.add_argument("--shift-z", type=float, default=0, help="shift along z-axis in the output particles(voxel units, default 0)")
    p.add_argument("--out-prefix", default=None, help="output filename prefix (default: tomogram stem in same directory)")
    p.add_argument("--debug", action="store_true", help="save intermediate MRC files for each pipeline step")
    p.add_argument("--recursive", action="store_true", help="process multiple tomograms (glob/directory)")
    args = p.parse_args()

    tomo_paths = _resolve_paths(args.tomo, recursive=args.recursive)
    multi = len(tomo_paths) > 1
    if multi:
        print(f"matched {len(tomo_paths)} tomograms")

    n_ok = n_fail = 0
    t_start = time.time()
    for idx, tomo_path in enumerate(tomo_paths, 1):
        print("=" * 88)
        print(f"[{idx}/{len(tomo_paths)}] {tomo_path}")
        out_prefix = _resolve_out_prefix(args, tomo_path, multi)
        try:
            _process_single_tomogram(args, tomo_path, out_prefix)
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            print(f"ERROR: {exc}")
            traceback.print_exc()

    print("=" * 88)
    print(f"done: {n_ok} ok, {n_fail} failed, total {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
