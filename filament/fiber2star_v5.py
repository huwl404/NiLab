#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mrcfile
import starfile
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation as R

import cupy as cp
from cupyx.scipy.ndimage import convolve as cp_convolve
from cucim import skimage as cskimage
from cucim.core.operations import intensity as cintensity

from skimage.morphology import skeletonize  # not implemented in cucim/cupy yet


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
            pass


def normalize_percentile(
    vol: np.ndarray, low: float = 1.0, high: float = 99.0
) -> np.ndarray:
    """GPU percentile normalization to [0,1]."""
    v_gpu = cp.asarray(vol, dtype=cp.float32)
    lo_gpu, hi_gpu = cp.percentile(v_gpu, [low, high])
    lo, hi = float(lo_gpu), float(hi_gpu)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    norm_gpu = cintensity.normalize_data(v_gpu, 1.0, lo, hi, type="range")
    norm_gpu = cp.clip(norm_gpu, 0.0, 1.0)
    return cp.asnumpy(norm_gpu).astype(np.float32)


def butterworth_preprocess(
    vol: np.ndarray,
    cutoff_frequency_ratio: float = 0.005,
    high_pass: bool = True,
    order: float = 2.0,
    squared_butterworth: bool = True,
) -> np.ndarray:
    """3D Butterworth filter on GPU."""
    v_gpu = cp.asarray(vol, dtype=cp.float32)
    out = cskimage.filters.butterworth(
        v_gpu,
        cutoff_frequency_ratio=cutoff_frequency_ratio,
        high_pass=high_pass,
        order=order,
        channel_axis=None,
        squared_butterworth=squared_butterworth,
        npad=0,
    )
    return cp.asnumpy(out).astype(np.float32)


def detect_membrane_sheetness(
    vol: np.ndarray,
    sigma: float = 2.0,
    sheetness_percentile: float = 95.0,
) -> np.ndarray:
    """3D Hessian-based sheet-like structure enhancement for membrane detection.
    Returns binary mask (1 = membrane candidate)."""
    v_gpu = cp.asarray(vol, dtype=cp.float32)
    H_elems = cskimage.feature.hessian_matrix(v_gpu, sigma=sigma, use_gaussian_derivatives=False)
    eigvals = cskimage.feature.hessian_matrix_eigvals(H_elems)
    lam1, lam2, lam3 = eigvals[0], eigvals[1], eigvals[2]
    Rb = cp.abs(lam2) / (cp.abs(lam1) + 1e-8)
    S = cp.sqrt(lam1**2 + lam2**2 + lam3**2 + 1e-8)
    alpha, beta = 0.4, 0.5 * cp.max(S)
    sheetness = cp.exp(-(Rb**2) / (2 * alpha**2)) * (1.0 - cp.exp(-(S**2) / (2 * beta**2)))
    sheetness = cp.where(lam1 > 0, sheetness, 0.0)
    pos = sheetness > 0
    if int(cp.sum(pos)) > 0:
        th = float(cp.percentile(sheetness[pos], sheetness_percentile))
    else:
        th = 0.0
    return cp.asnumpy((sheetness >= th).astype(cp.float32)).astype(np.float32)


def clean_membrane_mask(
    membrane_mask: np.ndarray,
    min_voxels: int = 10000,
    border_width: int = 4,
    min_inplane_width: float = 2.0,
) -> np.ndarray:
    """Clean membrane mask: remove border, small CCs, and narrow strips (e.g. microtubule walls)."""
    mem_gpu = cp.asarray(membrane_mask > 0.5)

    if border_width > 0:
        bw = int(border_width)
        mem_gpu[:bw, :, :] = False
        mem_gpu[-bw:, :, :] = False
        mem_gpu[:, :bw, :] = False
        mem_gpu[:, -bw:, :] = False
        mem_gpu[:, :, :bw] = False
        mem_gpu[:, :, -bw:] = False

    mem_gpu = cskimage.morphology.remove_small_objects(mem_gpu.astype(bool), min_size=int(min_voxels), connectivity=1)
    if not mem_gpu.any():
        return cp.asnumpy(mem_gpu).astype(np.float32)

    labels_gpu = cskimage.measure.label(mem_gpu.astype(cp.uint8), connectivity=1)
    num_labels = int(labels_gpu.max().get())

    for lab in range(1, num_labels + 1):
        mask_lab = labels_gpu == lab
        if not bool(mask_lab.any().get()):
            continue
        coords = cp.argwhere(mask_lab)
        if coords.shape[0] < 5:
            mem_gpu[mask_lab] = False
            continue
        coords_f = coords.astype(cp.float32)
        mean = coords_f.mean(axis=0, keepdims=True)
        X = coords_f - mean
        C = (X.T @ X) / (coords_f.shape[0] - 1.0)
        evals, evecs = cp.linalg.eigh(C)
        order = cp.argsort(cp.abs(evals))[::-1]
        evecs = evecs[:, order]
        v1, v2 = evecs[:, 0], evecs[:, 1]
        p1, p2 = X @ v1, X @ v2
        len1 = float(p1.max().get() - p1.min().get())
        len2 = float(p2.max().get() - p2.min().get())
        if min(len1, len2) < min_inplane_width:
            mem_gpu[mask_lab] = False

    return cp.asnumpy(mem_gpu).astype(np.float32)


def build_nonmembrane_mask(
    membrane_mask: np.ndarray,
    dilate_pixels: int = 4,
    border_clear_pixels: int = 4,
) -> np.ndarray:
    """Build non-membrane mask: dilate membrane, invert, clear borders."""
    mem_gpu = cp.asarray(membrane_mask > 0.5)
    if dilate_pixels > 0:
        selem = cskimage.morphology.ball(int(dilate_pixels))
        mem_gpu = cskimage.morphology.binary_dilation(mem_gpu, selem)
    nonmem_gpu = (~mem_gpu).astype(bool)
    if border_clear_pixels > 0:
        bw = int(border_clear_pixels)
        z, y, x = nonmem_gpu.shape
        bw = min(bw, z // 2, y // 2, x // 2)
        if bw > 0:
            nonmem_gpu[:bw, :, :] = False
            nonmem_gpu[-bw:, :, :] = False
            nonmem_gpu[:, :bw, :] = False
            nonmem_gpu[:, -bw:, :] = False
            nonmem_gpu[:, :, :bw] = False
            nonmem_gpu[:, :, -bw:] = False
    return cp.asnumpy(nonmem_gpu).astype(np.float32)


def extract_binary_mask(
    convmap_zyx: np.ndarray,
    nonmembrane_mask: np.ndarray,
    convmap_percentile: float,
    min_fiber_voxels: int,
    voxel_size: float,
    debug_prefix: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Extract cleaned binary mask from convmap volume (NO skeletonization).

    Steps:
      1. Mask convmap with non-membrane region
      2. Threshold at percentile
      3. Remove small connected components

    Returns boolean mask (ZYX order), or None if empty.
    Skeletonization is deferred to skeleton_to_segments() after direction
    filtering, so that direction estimation and filtering operate on well-
    connected thick CCs rather than fragmented skeleton voxels.
    """
    v = np.asarray(convmap_zyx, dtype=np.float32)
    nm = np.asarray(nonmembrane_mask, dtype=np.float32)

    v_gpu = cp.asarray(v, dtype=cp.float32)
    nm_gpu = cp.asarray(nm > 0.5)
    v_gpu = v_gpu * nm_gpu.astype(cp.float32)

    finite_vals = v_gpu[v_gpu > 0]
    if finite_vals.size == 0:
        print("  convmap is all zero after membrane masking")
        return None

    th = float(cp.percentile(finite_vals, convmap_percentile))
    print(f"  {convmap_percentile}% percentile => threshold = {th:.6f}")
    mask_gpu = v_gpu >= th
    if not bool(mask_gpu.sum().get()):
        print("  mask is empty after thresholding")
        return None
    if debug_prefix is not None:
        save_mrc(debug_prefix.with_name(debug_prefix.name + "_6_convmap_binary.mrc"), cp.asnumpy(mask_gpu.astype(cp.float32)), voxel_size=voxel_size)

    print(f"  small objects removal: min_size = {max(1, min_fiber_voxels)}")
    mask_gpu = cskimage.morphology.remove_small_objects(mask_gpu, min_size=max(1, min_fiber_voxels), connectivity=1)
    if not bool(mask_gpu.sum().get()):
        print("  all connected components removed (too small)")
        return None
    if debug_prefix is not None:
        save_mrc(debug_prefix.with_name(debug_prefix.name + "_7_convmap_clean.mrc"), cp.asnumpy(mask_gpu.astype(cp.float32)), voxel_size=voxel_size)

    return cp.asnumpy(mask_gpu.astype(bool))


def compute_dominant_direction(
    mask_zyx: np.ndarray,
    min_pts_per_cc: int = 50,
) -> Optional[np.ndarray]:
    """Compute dominant fiber direction from binary mask CCs.

    Operates on thick binary masks (NOT skeletons) so that CCs are well-
    connected and few in number, giving reliable PCA directions.
    Returns length-weighted average direction as unit vector (x,y,z), or None.
    """
    mask_gpu = cp.asarray(mask_zyx.astype(bool))
    labels_gpu = cskimage.measure.label(mask_gpu.astype(cp.uint8), connectivity=2)
    num_labels = int(labels_gpu.max().get())
    if num_labels == 0:
        return None
    labels_np = cp.asnumpy(labels_gpu)

    coords_zyx = np.argwhere(labels_np > 0)
    labs = labels_np[coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]]
    sort_idx = np.argsort(labs)
    sorted_zyx = coords_zyx[sort_idx]
    sorted_labs = labs[sort_idx]
    boundaries = np.searchsorted(sorted_labs, np.arange(1, num_labels + 2))

    directions: List[np.ndarray] = []
    lengths: List[float] = []
    for lab in range(1, num_labels + 1):
        s, e = boundaries[lab - 1], boundaries[lab]
        if e - s < min_pts_per_cc:
            continue
        pts_zyx_cc = sorted_zyx[s:e]
        pts = np.stack([pts_zyx_cc[:, 2], pts_zyx_cc[:, 1], pts_zyx_cc[:, 0]], axis=1).astype(np.float32)
        centered = pts - pts.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            d = vh[0].astype(np.float32)
            proj = centered @ d
            length = float(proj.max() - proj.min())
        except Exception:
            continue
        directions.append(d)
        lengths.append(length)

    if not directions:
        return None
    ref_idx = int(np.argmax(lengths))
    ref_dir = directions[ref_idx]
    w = np.array(lengths, dtype=np.float64)
    d_arr = np.array(
        [d if np.dot(d, ref_dir) >= 0 else -d for d in directions],
        dtype=np.float64,
    )
    dominant = np.sum(w[:, None] * d_arr, axis=0)
    norm = np.linalg.norm(dominant)
    if norm < 1e-12:
        return None
    return (dominant / norm).astype(np.float32)


def skeleton_junction_filter(
    mask_zyx: np.ndarray,
    dominant_dir: Optional[np.ndarray],
    max_angle_deg: float = 45.0,
    bridge_radius: int = 3,
    junction_clearance: int = 2,
    min_branch_pts: int = 3,
    voxel_size: float = 1.0,
    debug_prefix: Optional[Path] = None,
) -> List[np.ndarray]:
    """Skeletonize → bridge → break at junctions → filter by direction → segments.

    Unlike erosion-based approaches, this precisely removes noise branches
    connected to correct fibers by detecting skeleton branch points (degree ≥ 3
    in 26-connectivity graph) and splitting there.

    Steps:
      1. Skeletonize binary mask → 1-voxel-wide skeleton
      2. Dilate to bridge micro-gaps, re-skeletonize for clean topology
      3. Detect branch points (skeleton voxels with ≥ 3 neighbors)
      4. Remove branch-point neighborhoods (radius = junction_clearance)
         → isolated sub-branches
      5. CC label branches → PCA direction → keep aligned with dominant_dir
         Branches < min_branch_pts voxels are dropped (noise fragments)
      6. Re-bridge kept branches → CC label → PCA-ordered segments
         (PCA ordering needed: np.argwhere gives raster-scan, not along-fiber)

    Returns list of (N,3) float32 arrays in (x,y,z) pixel coords.
    """
    # Step 1: Skeletonize
    skel_np = skeletonize(mask_zyx.astype(bool))
    if not skel_np.any():
        print("  skeleton is empty")
        return []
    if debug_prefix is not None:
        save_mrc(debug_prefix.with_name(debug_prefix.name + "_8_skeleton_raw.mrc"), skel_np.astype(np.float32), voxel_size=voxel_size)

    # Step 2: Bridge micro-gaps + re-skeletonize
    if bridge_radius > 0:
        skel_gpu = cp.asarray(skel_np)
        bridge_selem = cskimage.morphology.ball(int(bridge_radius))
        bridged_gpu = cskimage.morphology.binary_dilation(skel_gpu, bridge_selem)
        skel_np = skeletonize(cp.asnumpy(bridged_gpu).astype(bool))
        if not skel_np.any():
            print("  skeleton empty after bridge + re-thin")
            return []
        if debug_prefix is not None:
            save_mrc(debug_prefix.with_name(debug_prefix.name + "_9_skeleton_bridged.mrc"), skel_np.astype(np.float32), voxel_size=voxel_size)

    skel_gpu = cp.asarray(skel_np)
    # Direction filter via junction splitting
    if dominant_dir is not None and max_angle_deg < 90.0:
        # Step 3: Detect branch/junction points (degree ≥ 3 in 26-connectivity)
        kernel = cp.ones((3, 3, 3), dtype=cp.int32)
        kernel[1, 1, 1] = 0
        n_neighbors = cp_convolve(skel_gpu.astype(cp.int32), kernel, mode='constant', cval=0)
        junctions_gpu = skel_gpu & (n_neighbors >= 3)
        n_junc = int(junctions_gpu.sum().get())
        print(f"  detected {n_junc} junction voxels on skeleton")

        # Step 4: Remove junction neighborhoods → sub-branches
        if n_junc > 0:
            if junction_clearance > 0:
                clear_selem = cskimage.morphology.ball(int(junction_clearance))
                junction_zone = cskimage.morphology.binary_dilation(junctions_gpu, clear_selem)
            else:
                junction_zone = junctions_gpu
            branches_gpu = skel_gpu & (~junction_zone)
        else:
            branches_gpu = skel_gpu

        if debug_prefix is not None:
            save_mrc(debug_prefix.with_name(debug_prefix.name + "_10_branches.mrc"), cp.asnumpy(branches_gpu.astype(cp.float32)), voxel_size=voxel_size)

        # Step 5: CC label → PCA direction → filter
        labels_gpu = cskimage.measure.label(branches_gpu.astype(cp.uint8), connectivity=2)
        num_labels = int(labels_gpu.max().get())
        if num_labels == 0:
            return []
        labels_np = cp.asnumpy(labels_gpu)

        coords_zyx = np.argwhere(labels_np > 0)
        labs = labels_np[coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]]
        sort_idx = np.argsort(labs)
        sorted_zyx = coords_zyx[sort_idx]
        sorted_labs = labs[sort_idx]
        boundaries = np.searchsorted(sorted_labs, np.arange(1, num_labels + 2))

        cos_thresh = math.cos(math.radians(max_angle_deg))
        keep_mask = np.zeros(num_labels + 1, dtype=bool)
        n_kept, n_removed, n_small = 0, 0, 0

        for lab in range(1, num_labels + 1):
            s, e = boundaries[lab - 1], boundaries[lab]
            if e - s < min_branch_pts:
                n_small += 1
                continue
            pts_zyx_cc = sorted_zyx[s:e]
            pts = np.stack([pts_zyx_cc[:, 2], pts_zyx_cc[:, 1], pts_zyx_cc[:, 0]], axis=1).astype(np.float32)
            centered = pts - pts.mean(axis=0)
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                d = vh[0]
            except Exception:
                continue
            d_aligned = d if np.dot(d, dominant_dir) >= 0 else -d
            if abs(float(np.dot(d_aligned, dominant_dir))) >= cos_thresh:
                keep_mask[lab] = True
                n_kept += 1
            else:
                n_removed += 1

        filtered_np = keep_mask[labels_np].astype(bool)
        print(f"  junction split => {num_labels} branches: kept {n_kept}, removed {n_removed}, small (dropped) {n_small}")

        if not filtered_np.any():
            print("  all branches removed")
            return []
        if debug_prefix is not None:
            save_mrc(debug_prefix.with_name(debug_prefix.name + "_11_filtered_branches.mrc"), filtered_np.astype(np.float32), voxel_size=voxel_size)
    else:
        filtered_np = skel_np.astype(bool)
        if dominant_dir is None:
            print("  skipping direction filter")

    # Step 6: Group branches via dilation CC, but extract ONLY thin skeleton voxels.
    # Dilation reconnects kept branches at former junction gaps for CC grouping,
    # but segments must be built from the 1-voxel skeleton to avoid thick sampling.
    thin_skel = filtered_np.astype(bool)
    filt_gpu = cp.asarray(thin_skel)
    if bridge_radius > 0:
        reconnect_selem = cskimage.morphology.ball(bridge_radius)
        grouped_gpu = cskimage.morphology.binary_dilation(filt_gpu, reconnect_selem)
    else:
        grouped_gpu = filt_gpu

    final_labels_gpu = cskimage.measure.label(grouped_gpu.astype(cp.uint8), connectivity=2)
    num_final = int(final_labels_gpu.max().get())
    if num_final == 0:
        return []
    final_labels_np = cp.asnumpy(final_labels_gpu)

    thin_labels = final_labels_np.copy()
    thin_labels[~thin_skel] = 0

    all_zyx = np.argwhere(thin_labels > 0)
    all_labs = thin_labels[all_zyx[:, 0], all_zyx[:, 1], all_zyx[:, 2]]
    sort_idx = np.argsort(all_labs)
    sorted_zyx = all_zyx[sort_idx]
    sorted_labs = all_labs[sort_idx]
    boundaries = np.searchsorted(sorted_labs, np.arange(1, num_final + 2))

    segments: List[np.ndarray] = []
    for lab in range(1, num_final + 1):
        s, e = boundaries[lab - 1], boundaries[lab]
        if e - s < 2:
            continue
        pts_zyx_cc = sorted_zyx[s:e]
        pts_xyz = np.stack(
            [pts_zyx_cc[:, 2].astype(np.float32),
             pts_zyx_cc[:, 1].astype(np.float32),
             pts_zyx_cc[:, 0].astype(np.float32)],
            axis=1,
        )
        mean = pts_xyz.mean(axis=0)
        centered = pts_xyz - mean
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            main_dir = vh[0]
        except Exception:
            continue
        proj = centered @ main_dir
        order = np.argsort(proj)
        pts_sorted = pts_xyz[order]

        dedup: List[int] = [0]
        for i in range(1, len(pts_sorted)):
            if np.linalg.norm(pts_sorted[i] - pts_sorted[dedup[-1]]) >= 1.0:
                dedup.append(i)
        if dedup[-1] != len(pts_sorted) - 1:
            dedup.append(len(pts_sorted) - 1)
        pts_dedup = pts_sorted[dedup]
        if len(pts_dedup) < 2:
            continue
        segments.append(pts_dedup.astype(np.float32))

    print(f"  {num_final} final branches => {len(segments)} segments")
    return segments


# ==============================
# Segment merging (Struwwel-Tracer-inspired CFS fusion)
# ==============================


def _endpoint_direction(pts: np.ndarray, from_end: bool, n_pts: int = 5) -> np.ndarray:
    """Compute outward direction at a segment endpoint using last/first n_pts points."""
    if len(pts) < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    k = min(n_pts, len(pts))
    if from_end:
        sub = pts[-k:]
        d = sub[-1] - sub[0]
    else:
        sub = pts[:k]
        d = sub[0] - sub[-1]
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (d / n).astype(np.float32)


def merge_fiber_segments(
    segments: List[np.ndarray],
    max_gap_px: float,
    max_angle_deg: float,
    dominant_dir: Optional[np.ndarray] = None,
    max_bridge_angle_deg: float = 45.0,
    extension_px: float = 0.0,
) -> List[np.ndarray]:
    """
    Merge fiber segments with dominant-direction bridge validation and
    both-end eligibility check.

    Three compatibility conditions for an endpoint pair (pi→di, pj→dj):
      a) effective distance < max_gap_px (with optional extension)
      b) outward directions roughly anti-parallel (endpoint angle check)
      c) bridge vector (pi→pj) roughly aligned with dominant_dir

    A segment is eligible only if BOTH its endpoints have at least one
    compatible partner.  Greedy priority: best bridge alignment with
    dominant_dir → longest total segment arc length → shortest gap.

    Returns all segments (merged eligible + unchanged non-eligible).
    """
    if len(segments) <= 1:
        return [s.copy() for s in segments]

    segs = [s.copy() for s in segments]
    max_gap_sq = max_gap_px ** 2
    cos_ep = math.cos(math.radians(max_angle_deg))
    cos_bridge = math.cos(math.radians(max_bridge_angle_deg)) if dominant_dir is not None else 0.0

    changed = True
    while changed:
        changed = False
        n = len(segs)
        if n <= 1:
            break

        arc_lens = []
        for seg in segs:
            if len(seg) < 2:
                arc_lens.append(0.0)
            else:
                arc_lens.append(float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))))

        ep_data: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
        for i, seg in enumerate(segs):
            if len(seg) < 2:
                continue
            ep_data.append((i, 0, seg[0].copy(), _endpoint_direction(seg, from_end=False)))
            ep_data.append((i, 1, seg[-1].copy(), _endpoint_direction(seg, from_end=True)))

        has_partner: Dict[Tuple[int, int], bool] = {}
        # (bridge_cos, total_arc, eff_sq, si, ei, sj, ej)
        compat: List[Tuple[float, float, float, int, int, int, int]] = []

        for a in range(len(ep_data)):
            for b in range(a + 1, len(ep_data)):
                si, ei, pi, di = ep_data[a]
                sj, ej, pj, dj = ep_data[b]
                if si == sj:
                    continue

                gap_vec = pj - pi
                dist_sq = float(np.sum(gap_vec ** 2))
                eff_sq = dist_sq
                if extension_px > 0 and dist_sq > 0:
                    gn = gap_vec / (math.sqrt(dist_sq) + 1e-8)
                    ext_i = max(0.0, float(np.dot(di, gn))) * extension_px
                    ext_j = max(0.0, float(np.dot(-dj, gn))) * extension_px
                    reduced = max(0.0, math.sqrt(dist_sq) - ext_i - ext_j)
                    eff_sq = reduced ** 2
                if eff_sq > max_gap_sq:
                    continue

                if float(np.dot(di, dj)) > -cos_ep:
                    continue

                bridge_cos = 1.0
                if dominant_dir is not None and dist_sq > 1e-12:
                    bridge_dir = gap_vec / (math.sqrt(dist_sq) + 1e-8)
                    bridge_cos = abs(float(np.dot(bridge_dir, dominant_dir)))
                    if bridge_cos < cos_bridge:
                        continue

                has_partner[(si, ei)] = True
                has_partner[(sj, ej)] = True
                total_arc = arc_lens[si] + arc_lens[sj]
                compat.append((bridge_cos, total_arc, eff_sq, si, ei, sj, ej))

        eligible = {i for i in range(n)
                    if (i, 0) in has_partner and (i, 1) in has_partner}

        compat.sort(key=lambda x: (-x[0], -x[1], x[2]))

        best_pair = None
        for _, _, _, si, ei, sj, ej in compat:
            if si in eligible and sj in eligible:
                best_pair = (si, ei, sj, ej)
                break

        if best_pair is None:
            break

        si, ei, sj, ej = best_pair
        seg_i, seg_j = segs[si], segs[sj]

        if ei == 1 and ej == 0:
            merged = np.vstack([seg_i, seg_j])
        elif ei == 0 and ej == 1:
            merged = np.vstack([seg_j, seg_i])
        elif ei == 1 and ej == 1:
            merged = np.vstack([seg_i, seg_j[::-1]])
        else:
            merged = np.vstack([seg_i[::-1], seg_j])

        new_segs = [seg for k, seg in enumerate(segs) if k != si and k != sj]
        new_segs.append(merged)
        segs = new_segs
        changed = True

    return segs


def _draw_segments_to_volume(segments: List[np.ndarray], shape_zyx: Tuple[int, ...],) -> np.ndarray:
    """Rasterize fiber segments into a labeled volume for debug visualization."""
    vol = np.zeros(shape_zyx, dtype=np.float32)
    for idx, seg in enumerate(segments, 1):
        sampled = resample_curve_by_spacing(seg, spacing_px=0.5)
        for pt in sampled:
            x, y, z = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
            if 0 <= z < shape_zyx[0] and 0 <= y < shape_zyx[1] and 0 <= x < shape_zyx[2]:
                vol[z, y, x] = float(idx)
    return vol


def resample_curve_by_spacing(curve_xyz: np.ndarray, spacing_px: float) -> np.ndarray:
    """Resample a curve at fixed arc-length intervals (pixel units)."""
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
    """Spline-smooth a point sequence and blend with endpoint line by curvature.

    curvature in [0,1]:
      - 0.0 => straight line between endpoints
      - 1.0 => keep full spline shape
    """
    pts = np.asarray(points_xyz, dtype=np.float32)
    n = len(pts)
    if n < 2:
        return pts.copy()
    curve_points = max(2, int(curve_points))
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total < 1e-6:
        t = np.linspace(0.0, 1.0, curve_points, dtype=np.float32)
        out = pts[0][None, :] * (1.0 - t[:, None]) + pts[-1][None, :] * t[:, None]
        return out.astype(np.float32)

    curv = float(np.clip(curvature, 0.0, 1.0))
    # Keep a mild smoothing baseline for denoising; geometric bending is
    # controlled by the explicit line-vs-spline blend below.
    smooth_factor = max(1e-6, n * 0.15)
    uq = np.linspace(0.0, 1.0, curve_points, dtype=np.float32)

    try:
        us = s / total
        tck_x = UnivariateSpline(us, pts[:, 0], s=smooth_factor, k=min(3, n - 1))
        tck_y = UnivariateSpline(us, pts[:, 1], s=smooth_factor, k=min(3, n - 1))
        tck_z = UnivariateSpline(us, pts[:, 2], s=smooth_factor, k=min(3, n - 1))
        curve = np.vstack([tck_x(uq), tck_y(uq), tck_z(uq)]).T.astype(np.float32)
    except Exception:
        curve = np.empty((curve_points, 3), dtype=np.float32)
        for i in range(3):
            curve[:, i] = np.interp(uq, s / total, pts[:, i])

    # Make curvature effect explicit and sensitive: blend spline with straight
    # endpoint chord. Lower curvature preserves less bending.
    line = pts[0][None, :] * (1.0 - uq[:, None]) + pts[-1][None, :] * uq[:, None]
    curve = line + curv * (curve - line)

    curve[0] = pts[0]
    curve[-1] = pts[-1]
    return curve.astype(np.float32)


def write_star(out_path: Path, coords_xyz: np.ndarray, angles: np.ndarray) -> None:
    """Write particle coordinates + Euler angles to STAR file (pixel coords, degrees)."""
    c = np.asarray(coords_xyz, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float32)
    df = pd.DataFrame(
        {
            "rlnCoordinateX": c[:, 0],
            "rlnCoordinateY": c[:, 1],
            "rlnCoordinateZ": c[:, 2],
            "rlnAngleRot": angles[:, 0],
            "rlnAngleTilt": angles[:, 1],
            "rlnAnglePsi": angles[:, 2],
        }
    )
    starfile.write({"0": df}, out_path, overwrite=True)


def vector_to_euler_zyz(vec_xyz: np.ndarray) -> Tuple[float, float, float]:
    """Convert a 3D direction vector (xyz array order) to RELION ZYZ Euler angles.

    We solve a rotation that maps reference Z axis [0,0,1] onto vec_xyz,
    then convert that rotation into intrinsic ZYZ Euler angles
    (rlnAngleRot, rlnAngleTilt, rlnAnglePsi).
    Returns (rot, tilt, psi) in degrees.
    """
    v = np.asarray(vec_xyz, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-6:
        return 0.0, 0.0, 0.0
    v = v / n
    z_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    try:
        rot_obj, _ = R.align_vectors(v[None, :], z_ref[None, :])
    except Exception:
        # Fallback: explicit shortest-arc rotation from z_ref to v.
        axis = np.cross(z_ref, v)
        s = float(np.linalg.norm(axis))
        c = float(np.clip(np.dot(z_ref, v), -1.0, 1.0))
        if s < 1e-8:
            if c > 0.0:
                rot_obj = R.identity()
            else:
                rot_obj = R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0], dtype=np.float64))
        else:
            axis /= s
            kx, ky, kz = axis
            K = np.array(
                [[0.0, -kz, ky],
                 [kz, 0.0, -kx],
                 [-ky, kx, 0.0]],
                dtype=np.float64,
            )
            rot_mat = np.eye(3, dtype=np.float64) + K * s + (K @ K) * (1.0 - c)
            rot_obj = R.from_matrix(rot_mat)
    ang = rot_obj.as_euler("ZYZ", degrees=True)
    return float(ang[0]), float(ang[1]), float(ang[2])


def estimate_tangents(points_xyz: np.ndarray, dominant_dir: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate per-point unit tangents and enforce consistent tangent sign."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    tang = np.zeros((n, 3), dtype=np.float64)
    if n == 1:
        tang[0] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        tang[0] = pts[1] - pts[0]
        tang[-1] = pts[-1] - pts[-2]
        if n > 2:
            tang[1:-1] = 0.5 * (pts[2:] - pts[:-2])

    norms = np.linalg.norm(tang, axis=1)
    valid = norms > 1e-8
    if not np.any(valid):
        tang[:] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return tang.astype(np.float32)

    tang[valid] /= norms[valid, None]
    first_valid = int(np.argmax(valid))
    for i in range(first_valid - 1, -1, -1):
        tang[i] = tang[i + 1]
    for i in range(first_valid + 1, n):
        if not valid[i]:
            tang[i] = tang[i - 1]

    if dominant_dir is not None:
        d = np.asarray(dominant_dir, dtype=np.float64).ravel()
        dn = np.linalg.norm(d)
        if np.isfinite(dn) and dn > 1e-8:
            d = d / dn
            for i in range(n):
                if np.dot(tang[i], d) < 0.0:
                    tang[i] = -tang[i]
            return tang.astype(np.float32)

    # No dominant direction: keep orientation continuous along the curve.
    for i in range(1, n):
        if np.dot(tang[i], tang[i - 1]) < 0.0:
            tang[i] = -tang[i]
    return tang.astype(np.float32)


# ==============================
# Main pipeline
# ==============================


def _looks_like_regex_path(path_pattern: str) -> bool:
    """Heuristic to detect regex-like path patterns."""
    # Typical regex tokens that are not plain filename characters.
    return bool(re.search(r"(\\.|\\d|\\D|\\w|\\W|\\s|\\S|[\^\$\+\?\|\(\)\[\]\{\}])", path_pattern))


def _resolve_regex_matches(path_pattern: str) -> List[Path]:
    """Resolve files by regex basename under a literal parent directory."""
    raw = Path(path_pattern).expanduser()
    parent_raw = raw.parent
    name_pattern = raw.name
    parent_dir = (parent_raw if parent_raw.is_absolute() else (Path.cwd() / parent_raw)).resolve()

    if not parent_dir.exists() or not parent_dir.is_dir():
        raise SystemExit(f"error: regex parent directory not found: {parent_dir}")
    try:
        name_re = re.compile(name_pattern)
    except re.error as exc:
        raise SystemExit(f"error: invalid regex in --tomo: {exc}")

    return [p.resolve() for p in parent_dir.rglob("*.mrc") if name_re.fullmatch(p.name)]


def _resolve_tomogram_paths(tomo_arg: str, recursive: bool) -> List[Path]:
    """Resolve input tomogram paths from single path / glob / regex."""
    raw = Path(tomo_arg).expanduser()
    if not recursive:
        tomo_path = (raw if raw.is_absolute() else (Path.cwd() / raw)).resolve()
        if not tomo_path.exists():
            raise SystemExit(f"error: tomogram not found: {tomo_path}")
        if not tomo_path.is_file():
            raise SystemExit(f"error: --tomo must be a file when --recursive is not set: {tomo_path}")
        return [tomo_path]

    # Recursive mode:
    # 1) wildcard path (e.g. test/cilia_*.mrc)
    # 2) regex path (e.g. test/cilia_\d+\.mrc)
    # 3) directory path (scan all *.mrc recursively)
    # 4) single file path
    if has_magic(tomo_arg):
        pattern = str(raw if raw.is_absolute() else (Path.cwd() / raw))
        matched = [Path(p).resolve() for p in glob(pattern, recursive=False)]
    elif _looks_like_regex_path(tomo_arg):
        matched = _resolve_regex_matches(tomo_arg)
    else:
        target = (raw if raw.is_absolute() else (Path.cwd() / raw)).resolve()
        if target.is_dir():
            matched = [p.resolve() for p in target.rglob("*.mrc")]
        elif target.exists():
            matched = [target]
        else:
            raise SystemExit(f"error: tomogram path not found: {target}")

    paths = sorted(matched)
    if not paths:
        raise SystemExit(f"error: no matched files found for --tomo={tomo_arg} with --recursive")
    return paths


def _resolve_out_prefix(args: argparse.Namespace, tomo_path: Path, multi_mode: bool) -> Path:
    """Build output prefix for one tomogram."""
    if args.out_prefix is not None:
        out_prefix = Path(args.out_prefix)
        if not out_prefix.is_absolute():
            out_prefix = tomo_path.parent / out_prefix
        if multi_mode:
            out_prefix = out_prefix.with_name(f"{out_prefix.name}_{tomo_path.stem}")
        return out_prefix
    return tomo_path.parent / tomo_path.stem


def _process_single_tomogram(args: argparse.Namespace, tomo_path: Path, out_prefix: Path) -> None:
    voxel = float(args.voxel_size)
    # ---- Load ----
    print(f"loading tomogram: {tomo_path}")
    vol = load_mrc(tomo_path)
    print(f"  volume shape (z,y,x): {vol.shape}")
    convmap_path = tomo_path.with_name(tomo_path.stem + "_convmap.mrc")
    if convmap_path.exists():
        print(f"loading convmap: {convmap_path}")
        convmap = load_mrc(convmap_path)
        print(f"  convmap shape (z,y,x): {convmap.shape}")
    else:
        raise SystemExit(f"error: convmap not found: {convmap_path}")

    # ---- Bin ----
    bin_factor = max(1, int(args.bin))
    if bin_factor > 1:
        print(f"binning by factor {bin_factor}")
        vol_gpu = cp.asarray(vol, dtype=cp.float32)
        convmap_gpu = cp.asarray(convmap, dtype=cp.float32)
        vol_gpu = cskimage.transform.downscale_local_mean(vol_gpu, (bin_factor, bin_factor, bin_factor))
        convmap_gpu = cskimage.transform.downscale_local_mean(convmap_gpu, (bin_factor, bin_factor, bin_factor))
        vol = cp.asnumpy(vol_gpu).astype(np.float32)
        convmap = cp.asnumpy(convmap_gpu).astype(np.float32)
        voxel *= float(bin_factor)
        print(f"  tomogram shape: {vol.shape}, convmap shape: {convmap.shape}, effective voxel: {voxel:.4f} A")
        if args.debug:
            save_mrc(out_prefix.with_name(out_prefix.name + "_0_binned_tomogram.mrc"), vol, voxel_size=voxel)
            save_mrc(out_prefix.with_name(out_prefix.name + "_0_binned_convmap.mrc"), convmap, voxel_size=voxel)

    # ---- Normalize ----
    print("normalizing tomogram: low = 1.0, high = 99.0")
    norm = normalize_percentile(vol, low=1.0, high=99.0)
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_1_norm.mrc"), norm, voxel_size=voxel)

    # ---- Butterworth low-pass ----
    cutoff_len_A = float(args.butterworth_cutoff)
    if cutoff_len_A > 0:
        cutoff_ratio = max(1e-4, min(voxel / cutoff_len_A, 0.49))
        print(f"Butterworth low-pass: cutoff = {cutoff_len_A} A, ratio = {cutoff_ratio:.6f}")
        filtered = butterworth_preprocess(norm, cutoff_frequency_ratio=cutoff_ratio, high_pass=False)
    else:
        filtered = norm
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_2_butterworth.mrc"), filtered, voxel_size=voxel)

    # ---- Membrane detection → non-membrane mask ----
    print(f"detecting membrane: sigma = {args.sigma}, percentile = {args.sheetness_percentile}%")
    membrane_mask = detect_membrane_sheetness(filtered, sigma=float(args.sigma), sheetness_percentile=float(args.sheetness_percentile))
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_3_membrane_mask_raw.mrc"), membrane_mask, voxel_size=voxel)

    print(f"cleaning membrane: min_voxels = {args.min_membrane_voxels}, min_inplane_width = {args.min_inplane_width}")
    membrane_mask = clean_membrane_mask(membrane_mask, min_voxels=max(1, int(args.min_membrane_voxels)), min_inplane_width=max(1, int(args.min_inplane_width)))
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_4_membrane_mask_clean.mrc"), membrane_mask, voxel_size=voxel)

    print(f"building non-membrane mask: dilate_pixels = 4, border_clear_pixels = 4")
    nonmembrane_mask = build_nonmembrane_mask(membrane_mask, dilate_pixels=4, border_clear_pixels=4)
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_5_nonmembrane_mask.mrc"), nonmembrane_mask, voxel_size=voxel)

    # ---- Extract binary mask at base percentile ----
    print(f"extracting binary mask in convmap: percentile = {args.convmap_percentile:.2f}% , min_voxels = {args.min_fiber_voxels}")
    base_mask = extract_binary_mask(
        convmap, nonmembrane_mask,
        convmap_percentile=float(args.convmap_percentile),
        min_fiber_voxels=int(args.min_fiber_voxels),
        voxel_size=voxel,
        debug_prefix=out_prefix if args.debug else None,
    )
    if base_mask is None:
        print("no binary mask extracted; nothing to do")
        return
    vol_shape = base_mask.shape

    # ---- Determine dominant direction from high-percentile reference mask ----
    ref_pct = float(args.ref_convmap_percentile)
    base_pct = float(args.convmap_percentile)
    dominant_dir = None
    while ref_pct > base_pct:
        print(f"  extracting binary mask in convmap: percentile = {ref_pct:.2f}% , min_voxels = 1")
        ref_mask = extract_binary_mask(
            convmap, nonmembrane_mask,
            convmap_percentile=ref_pct,
            min_fiber_voxels=1,
            voxel_size=voxel,
            debug_prefix=None,
        )
        if ref_mask is not None:
            if args.debug:
                save_mrc(out_prefix.with_name(out_prefix.name + "_7r_convmap_refmask.mrc"), ref_mask.astype(np.float32), voxel_size=voxel)
            break
            # ref_gpu = cp.asarray(ref_mask.astype(bool))
            # n_ccs = int(cskimage.measure.label(ref_gpu.astype(cp.uint8), connectivity=2).max().get())
            # print(f"    {n_ccs} CCs in reference mask")
            # if n_ccs >= 3:
            #     dominant_dir = compute_dominant_direction(ref_mask)
            #     if dominant_dir is not None:
            #         print(f"    dominant direction from ref: [{dominant_dir[0]:.3f}, {dominant_dir[1]:.3f}, {dominant_dir[2]:.3f}]")
            #         break
        ref_pct -= 0.1

    # if dominant_dir is None:
    #     print("  falling back to dominant direction from base mask")
    #     dominant_dir = compute_dominant_direction(base_mask)
    #     if dominant_dir is not None:
    #         print(f"    dominant direction: [{dominant_dir[0]:.3f}, {dominant_dir[1]:.3f}, {dominant_dir[2]:.3f}]")
    #     else:
    #         print("  WARNING: could not determine dominant direction; skipping direction filter")

    # # ---- Skeletonize → junction break → direction filter → segments ----
    # print("skeleton junction filter")
    # segments_raw = skeleton_junction_filter(
    #     base_mask, dominant_dir,
    #     max_angle_deg=float(args.direction_filter_angle),
    #     bridge_radius=int(args.skel_bridge_radius),
    #     voxel_size=voxel,
    #     debug_prefix=out_prefix if args.debug else None,
    # )
    # if not segments_raw:
    #     print("no segments found; nothing to do")
    #     return
    # if args.debug:
    #     save_mrc(out_prefix.with_name(out_prefix.name + "_12_segments.mrc"), _draw_segments_to_volume(segments_raw, vol_shape), voxel_size=voxel)

    # # ---- Merge segments ----
    # merge_gap_px = float(args.merge_gap) / voxel
    # merge_extend_px = float(args.merge_extend) / voxel
    # print(f"merging segments (gap:{args.merge_gap} Å = {merge_gap_px:.1f} px, angle: {args.merge_angle}°, extend: {args.merge_extend} Å = {merge_extend_px:.1f} px)")
    # fibers = merge_fiber_segments(
    #     segments_raw,
    #     max_gap_px=merge_gap_px,
    #     max_angle_deg=float(args.merge_angle),
    #     dominant_dir=dominant_dir,
    #     max_bridge_angle_deg=float(args.direction_filter_angle),
    #     extension_px=merge_extend_px,
    # )
    # print(f"  {len(segments_raw)} segments => {len(fibers)} fibers after merging")
    # if args.debug:
    #     save_mrc(out_prefix.with_name(out_prefix.name + "_13_fibers_merged.mrc"), _draw_segments_to_volume(fibers, vol_shape), voxel_size=voxel)

    # # ---- Length filter ----
    # min_len_px = float(args.min_fiber_length) / voxel
    # fibers = [f for f in fibers if float(np.sum(np.linalg.norm(np.diff(f, axis=0), axis=1))) >= min_len_px]
    # print(f"  length filter (>= {args.min_fiber_length} A = {min_len_px:.1f} px): {len(fibers)} fibers kept")
    # if not fibers:
    #     print("no fibers remain after length filter; nothing to do")
    #     return
    # if args.debug:
    #     save_mrc(out_prefix.with_name(out_prefix.name + "_14_fibers_filtered.mrc"), _draw_segments_to_volume(fibers, vol_shape), voxel_size=voxel)

    # # ---- Smooth and export ----
    # spacing_px = float(args.spacing) / voxel
    # all_coords: List[np.ndarray] = []
    # all_angles: List[np.ndarray] = []

    # for idx, fiber in enumerate(fibers):
    #     arc_len = float(np.sum(np.linalg.norm(np.diff(fiber, axis=0), axis=1)))
    #     curve_pts = max(20, int(arc_len / 3.0))
    #     curv = float(np.clip(args.curvature, 0.0, 1.0))
    #     smoothed = smooth_curve(fiber, curve_points=curve_pts, curvature=curv)

    #     pts = resample_curve_by_spacing(smoothed, spacing_px=spacing_px)
    #     if len(pts) == 0:
    #         continue

    #     tangents = estimate_tangents(pts, dominant_dir=dominant_dir)
    #     angs = np.zeros((len(pts), 3), dtype=np.float32)
    #     for i, t in enumerate(tangents):
    #         rot, tilt, psi = vector_to_euler_zyz(t)
    #         angs[i] = np.array([rot, tilt, psi], dtype=np.float32)
    #     all_coords.append(pts.astype(np.float32))
    #     all_angles.append(angs)

    # if not all_coords:
    #     print("no particles sampled; STAR will not be written")
    # else:
    #     coords = np.vstack(all_coords).astype(np.float32)
    #     angles = np.vstack(all_angles).astype(np.float32)
    #     star_path = out_prefix.with_name(out_prefix.name + "_particles.star")
    #     write_star(star_path, coords, angles)
    #     print(f"saved {len(coords)} particles in {star_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fiber tracking in cryo-ET tomograms: Sato vesselness → candidate segments → directional merging → .star output.")
    parser.add_argument("--tomo", required=True, help="input .mrc path; with --recursive supports glob (test/cilia_*.mrc) or regex basename (test/cilia_\\d+\\.mrc)")
    parser.add_argument("--recursive", action="store_true", help="process multiple tomograms (wildcard or directory) recursively")
    parser.add_argument("--voxel-size", type=float, required=True, help="voxel size in Angstrom (required)")
    parser.add_argument("--bin", type=int, default=2, help="integer binning factor (default 2)")
    # Butterworth preprocessing
    parser.add_argument("--butterworth-cutoff", type=float, default=300.0, help="Butterworth low-pass cutoff (Angstrom, default 300)")
    # Membrane detection
    parser.add_argument("--sigma", type=float, default=1.0, help="sigma for membrane sheetness (default 1.0)")
    parser.add_argument("--sheetness-percentile", type=float, default=94.0, help="percentile threshold for membrane sheetness (default 94)")
    parser.add_argument("--min-membrane-voxels", type=int, default=1000, help="minimum voxels for membrane cleaning (default 1000)")
    parser.add_argument("--min-inplane-width", type=int, default=10, help="minimum inplane width for membrane cleaning (default 10 voxels)")
    # Candidate segment extraction
    parser.add_argument("--convmap-percentile", type=float, default=99.0, help="percentile threshold on convmap for binary mask (default 99)")
    parser.add_argument("--min-fiber-voxels", type=int, default=10, help="minimum voxel count for candidate CCs (default 10)")
    parser.add_argument("--skel-bridge-radius", type=int, default=2, help="dilation radius to bridge skeleton gaps before size filtering (default 2)")
    parser.add_argument("--direction-filter-angle", type=float, default=45.0, help="max angle from dominant fiber direction (degrees, default 45; >=90 disables)")
    parser.add_argument("--ref-convmap-percentile", type=float, default=99.99, help="convmap percentile for reference mask used to determine dominant fiber direction (default 99.99; auto-lowers by 0.01 until >= 3 CCs found)")
    # Segment merging (Struwwel-Tracer-inspired)
    parser.add_argument("--merge-gap", type=float, default=500.0, help="max gap for segment merging (Angstrom, default 400)")
    parser.add_argument("--merge-angle", type=float, default=45.0, help="max angle deviation for segment merging (degrees, default 45)")
    parser.add_argument("--merge-extend", type=float, default=100.0, help="virtual extension at endpoints for gap bridging (Angstrom, default 100)")
    parser.add_argument("--min-fiber-length", type=float, default=400.0, help="minimum fiber arc length after merging (Angstrom, default 400)")
    # Output
    parser.add_argument("--curvature", type=float, default=0.2, help="curve bending control [0,1]: 0=straight endpoint line, 1=full spline shape (default 0.2)")
    parser.add_argument("--spacing", type=float, default=40.0, help="particle sampling spacing along fibers (Angstrom, default 40)")
    parser.add_argument("--out-prefix", default=None, help="output file prefix")
    parser.add_argument("--debug", action="store_true", help="enable debug intermediate outputs")
    args = parser.parse_args()

    if float(args.voxel_size) <= 0:
        print("error: --voxel-size must be > 0")
        return

    tomo_paths = _resolve_tomogram_paths(args.tomo, recursive=bool(args.recursive))
    multi_mode = len(tomo_paths) > 1

    if multi_mode:
        print(f"matched {len(tomo_paths)} tomograms for processing")
        if args.out_prefix is not None:
            print("note: --out-prefix is automatically suffixed by each tomogram stem in multi-file mode")

    n_ok = 0
    n_fail = 0
    time_start = time.time()
    for idx, tomo_path in enumerate(tomo_paths, 1):
        print("=" * 88)
        print(f"[{idx}/{len(tomo_paths)}] processing: {tomo_path}")
        out_prefix = _resolve_out_prefix(args, tomo_path, multi_mode=multi_mode)
        try:
            _process_single_tomogram(args, tomo_path, out_prefix)
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            print(f"error while processing {tomo_path}: {exc}")

    print("=" * 88)
    print(f"done: {n_ok} succeeded, {n_fail} failed, in {time.time() - time_start:.2f}s")


if __name__ == "__main__":
    main()
