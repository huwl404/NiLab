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
  9. Segment merging         : greedy nearest-endpoint merge (gap + angle check)
 10. Length filter
 11. Line-fit + sample       : nearly-straight fibers → PCA line → evenly spaced pts
 12. Write .star              : rlnCoordinateX/Y/Z + rlnAngleRot/Tilt/Psi
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
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import mrcfile
import starfile
from scipy.spatial.transform import Rotation as R
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _sp_cc

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian1d
from cupyx.scipy.ndimage import convolve as cp_convolve
from cucim import skimage as cskimage

from skimage.morphology import skeletonize  # not yet in cucim
from skimage import io as skio


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


def write_star(out_path: Path, coords_xyz: np.ndarray, angles: np.ndarray) -> None:
    """Write RELION-style particle STAR (pixel coords, ZYZ Euler degrees)."""
    c = np.asarray(coords_xyz, dtype=np.float32)
    a = np.asarray(angles, dtype=np.float32)
    df = pd.DataFrame({
        "rlnCoordinateX": c[:, 0],
        "rlnCoordinateY": c[:, 1],
        "rlnCoordinateZ": c[:, 2],
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
# Step 7b (replaces merge): lateral-proximity clustering of segments
# ─────────────────────────────────────────────────────────────────────────────


def group_segments_by_fiber(
    segments: List[np.ndarray],
    dominant_dir: np.ndarray,
    fiber_radius_px: float,
    max_angle_deg: float = 30.0,
    debug_prefix: Optional[Path] = None,
) -> List[np.ndarray]:
    """
    Build fiber chains from noisy skeleton segments with strict constraints:
      1) lateral distance (in plane ⟂ dominant_dir) <= fiber_radius_px
      2) forward-only links along dominant_dir
      3) gap direction must be close to dominant_dir (angle <= max_angle_deg)
      4) one predecessor + one successor per segment (one-to-one directed graph)
      5) greedy score prefers long segments first, then directional consistency
    """
    if not segments:
        return []

    d = dominant_dir / np.linalg.norm(dominant_dir)
    cos_thr = math.cos(math.radians(max_angle_deg))
    n = len(segments)
    max_gap_along = max(4.0 * float(fiber_radius_px), 12.0)

    def _safe_unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        nv = float(np.linalg.norm(v))
        if not np.isfinite(nv) or nv < 1e-8:
            return fallback.copy()
        return (v / nv).astype(np.float64)

    def _segment_dir(seg: np.ndarray) -> np.ndarray:
        seg64 = seg.astype(np.float64)
        if len(seg64) >= 3:
            try:
                _, _, vh = np.linalg.svd(seg64 - seg64.mean(0), full_matrices=False)
                vd = _safe_unit(vh[0], d)
            except Exception:
                vd = _safe_unit(seg64[-1] - seg64[0], d)
        else:
            vd = _safe_unit(seg64[-1] - seg64[0], d)
        if float(np.dot(vd, d)) < 0.0:
            vd = -vd
        return vd

    centroids = np.array([seg.mean(0) for seg in segments], dtype=np.float64)
    along_c = centroids @ d
    perp_c = centroids - along_c[:, None] * d
    seg_dirs = np.array([_segment_dir(seg) for seg in segments], dtype=np.float64)
    seg_lens = np.array(
        [float(np.sum(np.linalg.norm(np.diff(seg.astype(np.float64), axis=0), axis=1))) if len(seg) >= 2 else 0.0
         for seg in segments],
        dtype=np.float64,
    )

    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dv = centroids[j] - centroids[i]
            gap_along = float(np.dot(dv, d))
            if gap_along <= 1e-6 or gap_along > max_gap_along:
                continue
            lateral = float(np.linalg.norm(dv - gap_along * d))
            if lateral > float(fiber_radius_px):
                continue
            gap_norm = float(np.linalg.norm(dv))
            if gap_norm < 1e-6:
                continue
            cos_gap = gap_along / gap_norm
            if cos_gap < cos_thr:
                continue
            if abs(float(np.dot(seg_dirs[i], d))) < cos_thr or abs(float(np.dot(seg_dirs[j], d))) < cos_thr:
                continue

            long_priority = min(seg_lens[i], seg_lens[j])
            score = (
                5.0 * long_priority
                + 1.5 * (seg_lens[i] + seg_lens[j])
                + 100.0 * cos_gap
                - 2.0 * lateral
                - 0.75 * gap_along
            )
            edges.append((score, i, j, gap_along, lateral, cos_gap))

    succ = -np.ones(n, dtype=np.int32)
    pred = -np.ones(n, dtype=np.int32)
    chosen_edges: List[Tuple[int, int]] = []
    for _, i, j, _, _, _ in sorted(edges, key=lambda x: x[0], reverse=True):
        if succ[i] >= 0 or pred[j] >= 0:
            continue
        succ[i] = j
        pred[j] = i
        chosen_edges.append((i, j))

    visited = np.zeros(n, dtype=bool)
    chains: List[List[int]] = []
    starts = np.where(pred < 0)[0]
    starts = starts[np.argsort(along_c[starts])]
    for s in starts:
        if visited[s]:
            continue
        ch = [int(s)]
        visited[s] = True
        cur = int(s)
        while succ[cur] >= 0:
            nxt = int(succ[cur])
            if visited[nxt]:
                break
            ch.append(nxt)
            visited[nxt] = True
            cur = nxt
        chains.append(ch)
    for i in range(n):
        if not visited[i]:
            chains.append([int(i)])
            visited[i] = True

    results: List[np.ndarray] = []
    seg_to_chain = -np.ones(n, dtype=np.int32)
    for ci, ch in enumerate(chains):
        for si in ch:
            seg_to_chain[si] = int(ci)

        ch_perp = perp_c[np.asarray(ch, dtype=np.int32)]
        ch_center = np.median(ch_perp, axis=0)
        keep_idxs = []
        for si in ch:
            if float(np.linalg.norm(perp_c[si] - ch_center)) <= float(fiber_radius_px):
                keep_idxs.append(si)
        if not keep_idxs:
            continue

        chain_pts: List[np.ndarray] = []
        for si in keep_idxs:
            seg = segments[si].astype(np.float64)
            p = seg @ d
            seg_sorted = seg[np.argsort(p)]
            if chain_pts:
                prev = chain_pts[-1][-1]
                if np.linalg.norm(seg_sorted[0] - prev) > np.linalg.norm(seg_sorted[-1] - prev):
                    seg_sorted = seg_sorted[::-1]
            chain_pts.append(seg_sorted)

        pts = np.vstack(chain_pts)
        p_all = pts @ d
        pts = pts[np.argsort(p_all)]
        dedup = [0]
        for k in range(1, len(pts)):
            if np.linalg.norm(pts[k] - pts[dedup[-1]]) >= 0.8:
                dedup.append(k)
        if dedup[-1] != len(pts) - 1:
            dedup.append(len(pts) - 1)
        pts = pts[dedup]
        if len(pts) >= 2:
            results.append(pts.astype(np.float32))

    if debug_prefix is not None:
        _save_cluster_debug_2d(
            centroids=centroids,
            dominant_dir=d,
            chain_ids=seg_to_chain,
            edges=chosen_edges,
            seg_lens=seg_lens,
            out_png=debug_prefix.parent / (debug_prefix.name + "_9_clustered_2d.png"),
        )
    return results


def _save_cluster_debug_2d(
    centroids: np.ndarray,
    dominant_dir: np.ndarray,
    chain_ids: np.ndarray,
    edges: List[Tuple[int, int]],
    seg_lens: np.ndarray,
    out_png: Path,
) -> None:
    """
    Save 2D clustering diagnostics in plane perpendicular to dominant direction.
    Each segment centroid is color-coded by chain ID; chosen directed links are drawn.
    """
    if len(centroids) == 0:
        return

    d = dominant_dir.astype(np.float64)
    d = d / np.linalg.norm(d)

    # Use projected +Z as image Y axis (as close as possible to tomogram Z).
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = z_axis - float(np.dot(z_axis, d)) * d
    if float(np.linalg.norm(v)) < 1e-6:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = y_axis - float(np.dot(y_axis, d)) * d
    v /= max(float(np.linalg.norm(v)), 1e-8)
    u = np.cross(v, d)
    u /= max(float(np.linalg.norm(u)), 1e-8)

    uv = np.stack([centroids @ u, centroids @ v], axis=1)
    uv_min = uv.min(axis=0)
    uv_max = uv.max(axis=0)
    span = np.maximum(uv_max - uv_min, 1.0)

    pad = 40
    max_canvas = 1400.0
    scale = (max_canvas - 2.0 * pad) / max(float(max(span[0], span[1])), 1.0)
    w = int(max(2 * pad + 20, round(span[0] * scale + 2 * pad)))
    h = int(max(2 * pad + 20, round(span[1] * scale + 2 * pad)))
    xy = np.empty_like(uv)
    # Flip both axes to match tomogram viewer orientation expectations.
    xy[:, 0] = pad + (uv_max[0] - uv[:, 0]) * scale
    xy[:, 1] = pad + (uv_max[1] - uv[:, 1]) * scale

    img = np.full((h, w, 3), 255, dtype=np.uint8)

    def _draw_disk(cx: int, cy: int, rad: int, color: np.ndarray) -> None:
        r = max(1, int(rad))
        y0, y1 = max(0, cy - r), min(h - 1, cy + r)
        x0, x1 = max(0, cx - r), min(w - 1, cx + r)
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[yy[m], xx[m]] = color

    def _draw_line(x0: float, y0: float, x1: float, y1: float, color: np.ndarray) -> None:
        dist = max(abs(x1 - x0), abs(y1 - y0))
        n_step = max(int(dist) + 1, 2)
        for t in np.linspace(0.0, 1.0, n_step):
            x = int(round(x0 + t * (x1 - x0)))
            y = int(round(y0 + t * (y1 - y0)))
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = color

    # Draw selected links first
    for i, j in edges:
        _draw_line(xy[i, 0], xy[i, 1], xy[j, 0], xy[j, 1], np.array([90, 90, 90], dtype=np.uint8))

    n_chain = int(np.max(chain_ids)) + 1 if len(chain_ids) else 0
    rng = np.random.default_rng(12345)
    palette = rng.integers(25, 235, size=(max(n_chain, 1), 3), dtype=np.uint8)

    q50 = float(np.percentile(seg_lens, 50)) if len(seg_lens) else 1.0
    q95 = float(np.percentile(seg_lens, 95)) if len(seg_lens) else max(q50, 1.0)
    denom = max(q95 - q50, 1e-6)

    for i in range(len(centroids)):
        cid = int(chain_ids[i])
        if cid < 0:
            color = np.array([0, 0, 0], dtype=np.uint8)
        else:
            color = palette[cid % len(palette)]
        r = 2 + int(np.clip((seg_lens[i] - q50) / denom, 0.0, 1.0) * 5.0)
        _draw_disk(int(round(xy[i, 0])), int(round(xy[i, 1])), r, color)

    try:
        skio.imsave(str(out_png), img)
    except Exception as e:
        print(f"  warn: failed saving cluster 2D debug image: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Steps 9: Line-fit sampling + Euler angles
# ─────────────────────────────────────────────────────────────────────────────


def fit_line_and_sample(
    segment_xyz: np.ndarray,
    dominant_dir: np.ndarray,
    spacing_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth polyline arc-length sampling for a fiber track.
    Returns (coords (N,3) float32 xyz, global_dir (3,) float32 xyz).
    """
    pts = np.asarray(segment_xyz, dtype=np.float64)
    if len(pts) == 0:
        d0 = np.asarray(dominant_dir, dtype=np.float64)
        d0 /= max(np.linalg.norm(d0), 1e-8)
        return np.empty((0, 3), dtype=np.float32), d0.astype(np.float32)

    # Deduplicate tiny spatial repeats to avoid zero-length arc segments.
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= 1e-6:
            keep.append(i)
    pts = pts[keep]
    if len(pts) == 1:
        d0 = np.asarray(dominant_dir, dtype=np.float64)
        d0 /= max(np.linalg.norm(d0), 1e-8)
        return pts.astype(np.float32), d0.astype(np.float32)

    # Lightweight smoothing to suppress jagged NN skeleton oscillation.
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

    # Arc-length parameterization and uniform sampling.
    seg_vec = np.diff(pts_smooth, axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = float(cum[-1])
    if total_len < max(spacing_px * 0.5, 1e-6):
        mid = pts_smooth.mean(0, keepdims=True)
        v = pts_smooth[-1] - pts_smooth[0]
        nv = float(np.linalg.norm(v))
        if nv < 1e-8:
            v = np.asarray(dominant_dir, dtype=np.float64)
            nv = max(float(np.linalg.norm(v)), 1e-8)
        d = v / nv
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
            out[k] = pts_smooth[-1]
            continue
        den = max(cum[j + 1] - cum[j], 1e-8)
        a = (t - cum[j]) / den
        out[k] = (1.0 - a) * pts_smooth[j] + a * pts_smooth[j + 1]

    global_dir = out[-1] - out[0]
    ng = float(np.linalg.norm(global_dir))
    if ng < 1e-8:
        global_dir = np.asarray(dominant_dir, dtype=np.float64)
        ng = max(float(np.linalg.norm(global_dir)), 1e-8)
    global_dir /= ng
    if float(np.dot(global_dir, dominant_dir)) < 0:
        global_dir = -global_dir
    return out.astype(np.float32), global_dir.astype(np.float32)


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


# ─────────────────────────────────────────────────────────────────────────────
# Debug visualization
# ─────────────────────────────────────────────────────────────────────────────


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

    # ── 7. Lateral-proximity clustering → fiber groups ────────────────────────
    fiber_radius_px = float(args.fiber_diameter) / 2.0 / voxel
    print(f"fiber clustering: diameter={args.fiber_diameter} Å (radius={fiber_radius_px:.1f} px), angle≤{args.direction_filter_angle}°")
    fibers = group_segments_by_fiber(
        segments,
        dominant_dir=dominant_dir,
        fiber_radius_px=fiber_radius_px,
        max_angle_deg=args.direction_filter_angle,
        debug_prefix=out_prefix if args.debug else None,
    )
    print(f"  {len(segments)} segments → {len(fibers)} fiber clusters")
    if args.debug:
        save_mrc(out_prefix.parent / (out_prefix.name + "_9_clustered.mrc"), _draw_segments_to_volume(fibers, vol_shape, radius=2, bridge_gaps=True), voxel)

    # ── 8. Length filter ──────────────────────────────────────────────────────
    if args.min_fiber_length > 0:
        min_len_px = float(args.min_fiber_length) / voxel
        fibers_long: List[np.ndarray] = []
        for fiber in fibers:
            arc = float(np.sum(np.linalg.norm(np.diff(fiber, axis=0), axis=1)))
            if arc >= min_len_px:
                fibers_long.append(fiber)
        print(f"  length filter (≥{args.min_fiber_length} Å: {len(fibers)} → {len(fibers_long)} fibers")
        fibers = fibers_long
        if args.debug:
            save_mrc(out_prefix.parent / (out_prefix.name + "_10_length_filtered.mrc"), _draw_segments_to_volume(fibers, vol_shape, radius=2, bridge_gaps=True), voxel)

    if not fibers:
        print("no fibers remain; exiting")
        return

    # ── 9. Sample + write STAR ───────────────────────────────────────────────
    spacing_px = float(args.spacing) / voxel
    all_coords: List[np.ndarray] = []
    all_angles: List[np.ndarray] = []
    fallback_dir = dominant_dir if dominant_dir is not None else np.array([1., 0., 0.], np.float32)

    for fiber in fibers:
        pts, line_dir = fit_line_and_sample(fiber, fallback_dir, spacing_px)
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
    write_star(star_path, coords, angles)
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
    p.add_argument("--direction-filter-angle", type=float, default=30.0, help="max angle from dominant fiber direction for branch/CC filtering (default 30°)")
    p.add_argument("--bridge-radius", type=int, default=3, help="dilation radius to bridge skeleton micro-gaps (default 3)")
    p.add_argument("--fiber-diameter", type=float, default=500.0, help="fiber diameter in Å used as clustering radius (default 500)")
    p.add_argument("--min-fiber-length", type=float, default=800.0, help="discard fibers shorter than this arc length (Å, default 800)")
    p.add_argument("--spacing", type=float, default=40, help="particle sampling spacing along fiber (Å, default 40)")
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
