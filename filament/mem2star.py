#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import mrcfile
import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation as R

import cupy as cp
from cucim import skimage as cskimage
from cucim.core.operations import intensity as cintensity
import cupyx.scipy.ndimage as ndi

warnings.filterwarnings(
    action="ignore",
    message=r".*cupyx\.jit\.rawkernel is experimental.*",
    category=FutureWarning,
    module="cupyx.jit._interface",
)


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


def normalize_percentile(vol: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """GPU percentile normalization to [0, 1]."""
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
    cutoff_frequency_ratio: float,
    high_pass: bool = False,
    order: float = 2.0,
    squared_butterworth: bool = True,
) -> np.ndarray:
    """3D Butterworth filtering on GPU."""
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
    """
    Hessian sheetness-based membrane detection.
    Returns binary float mask (ZYX): 1 = membrane candidate.
    """
    v_gpu = cp.asarray(vol, dtype=cp.float32)
    h_elems = cskimage.feature.hessian_matrix(v_gpu, sigma=sigma, use_gaussian_derivatives=False)
    eigvals = cskimage.feature.hessian_matrix_eigvals(h_elems)
    lam1, lam2, lam3 = eigvals[0], eigvals[1], eigvals[2]

    rb = cp.abs(lam2) / (cp.abs(lam1) + 1e-8)
    s = cp.sqrt(lam1**2 + lam2**2 + lam3**2 + 1e-8)
    alpha, beta = 0.4, 0.5 * cp.max(s)
    sheetness = cp.exp(-(rb**2) / (2 * alpha**2)) * (1.0 - cp.exp(-(s**2) / (2 * beta**2)))
    sheetness = cp.where(lam1 > 0, sheetness, 0.0)

    valid = sheetness > 0
    th = float(cp.percentile(sheetness[valid], sheetness_percentile)) if int(cp.sum(valid)) > 0 else 0.0
    return cp.asnumpy((sheetness >= th).astype(cp.float32))


def clean_membrane_mask(
    membrane_mask: np.ndarray,
    min_voxels: int = 10000,
    border_width: int = 4,
    min_inplane_width: float = 20.0,
    ball_radius: Optional[int] = None,
) -> np.ndarray:
    """
    Clean membrane mask without dilation:
    - clear borders
    - remove tiny CCs
    - remove narrow strip-like CCs by in-plane PCA span
    """
    mem_gpu = cp.asarray(membrane_mask > 0.5)

    if border_width > 0:
        bw = int(border_width)
        mem_gpu[:bw, :, :] = False
        mem_gpu[-bw:, :, :] = False
        mem_gpu[:, :bw, :] = False
        mem_gpu[:, -bw:, :] = False
        mem_gpu[:, :, :bw] = False
        mem_gpu[:, :, -bw:] = False
    
    if ball_radius is not None and ball_radius > 0:
        selem = cskimage.morphology.ball(int(ball_radius))
        mem_gpu = cskimage.morphology.binary_dilation(mem_gpu, selem)

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
        centered = coords_f - coords_f.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(coords_f.shape[0] - 1.0, 1.0)
        evals, evecs = cp.linalg.eigh(cov)
        order = cp.argsort(cp.abs(evals))[::-1]
        evecs = evecs[:, order]
        p1 = centered @ evecs[:, 0]
        p2 = centered @ evecs[:, 1]
        len1 = float(p1.max().get() - p1.min().get())
        len2 = float(p2.max().get() - p2.min().get())
        if min(len1, len2) < float(min_inplane_width):
            mem_gpu[mask_lab] = False

    return cp.asnumpy(mem_gpu.astype(cp.float32))


def dilate_membrane_mask(membrane_clean: np.ndarray, dilate_radius_px: int) -> np.ndarray:
    """Step 1-2: dilate clean membrane mask."""
    mem_gpu = cp.asarray(membrane_clean > 0.5)
    r = max(0, int(dilate_radius_px))
    if r > 0:
        selem = cskimage.morphology.ball(r)
        mem_gpu = cskimage.morphology.binary_dilation(mem_gpu, selem)
    return cp.asnumpy(mem_gpu.astype(cp.float32))


def extract_potential_membrane_proteins_from_shell(
    membrane_with_ball: np.ndarray,
    membrane_clean: np.ndarray,
    min_voxels: int = 10,
    max_voxels: int = 100,
) -> np.ndarray:
    """
    Step 3:
      shell = dilated - clean
      keep connected components with size in [min_voxels, max_voxels]
    """
    dilated_gpu = cp.asarray(membrane_with_ball > 0.5)
    clean_gpu = cp.asarray(membrane_clean > 0.5)
    shell_gpu = dilated_gpu & (~clean_gpu)
    if not bool(shell_gpu.any()):
        return np.zeros(cp.asnumpy(shell_gpu).shape, dtype=bool)

    structure = cp.asarray(ndi.generate_binary_structure(3, 1))
    labels_gpu, nlab = ndi.label(shell_gpu, structure=structure)
    if nlab == 0:
        return np.zeros(cp.asnumpy(shell_gpu).shape, dtype=bool)

    sizes = cp.bincount(labels_gpu.ravel())
    keep = (sizes >= max(1, int(min_voxels))) & (sizes <= max(1, int(max_voxels)))
    keep[0] = False
    protein_mask_gpu = keep[labels_gpu].astype(cp.bool_)
    return cp.asnumpy(protein_mask_gpu)


def particles_with_convex_normals(
    protein_mask: np.ndarray,
    membrane_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert membrane-bound protein CCs to particles and normals.

    Normal definition:
      normal = particle_centroid - nearest_membrane_voxel
    This enforces z-axis pointing from membrane toward protein side (convex side).
    """
    mem_gpu = cp.asarray(membrane_mask > 0.5)
    prot_gpu = cp.asarray(protein_mask.astype(bool))
    if not bool(mem_gpu.any()) or not bool(prot_gpu.any()):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    _, nearest_idx_gpu = ndi.distance_transform_edt(~mem_gpu, return_indices=True)
    structure = cp.asarray(ndi.generate_binary_structure(3, 1))
    labels_gpu, nlab = ndi.label(prot_gpu, structure=structure)
    if nlab == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    labels = cp.asnumpy(labels_gpu)
    coords_xyz: List[np.ndarray] = []
    normals_xyz: List[np.ndarray] = []
    for lab in range(1, nlab + 1):
        pts = np.argwhere(labels == lab)
        if pts.size == 0:
            continue
        centroid_zyx = pts.mean(axis=0)
        cz, cy, cx = centroid_zyx

        pz = pts[:, 0].astype(np.int64)
        py = pts[:, 1].astype(np.int64)
        px = pts[:, 2].astype(np.int64)
        pz_gpu = cp.asarray(pz)
        py_gpu = cp.asarray(py)
        px_gpu = cp.asarray(px)
        nz = cp.asnumpy(nearest_idx_gpu[0, pz_gpu, py_gpu, px_gpu]).astype(np.float32)
        ny = cp.asnumpy(nearest_idx_gpu[1, pz_gpu, py_gpu, px_gpu]).astype(np.float32)
        nx = cp.asnumpy(nearest_idx_gpu[2, pz_gpu, py_gpu, px_gpu]).astype(np.float32)

        # Use mean vector from attached membrane surface to component voxels.
        vx = px.astype(np.float32) - nx
        vy = py.astype(np.float32) - ny
        vz = pz.astype(np.float32) - nz
        normal_xyz = np.array([float(np.mean(vx)), float(np.mean(vy)), float(np.mean(vz))], dtype=np.float32)
        nn = float(np.linalg.norm(normal_xyz))
        if nn < 1e-6:
            continue
        normal_xyz /= nn

        coords_xyz.append(np.array([cx, cy, cz], dtype=np.float32))
        normals_xyz.append(normal_xyz.astype(np.float32))

    if not coords_xyz:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.vstack(coords_xyz), np.vstack(normals_xyz)


def vector_to_euler_zyz(vec_xyz: np.ndarray) -> Tuple[float, float, float]:
    """Convert direction vector to RELION ZYZ Euler angles."""
    v = np.asarray(vec_xyz, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-8:
        return 0.0, 0.0, 0.0
    v /= n
    z_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rot_obj, _ = R.align_vectors(v[None, :], z_ref[None, :])
    ang = rot_obj.as_euler("ZYZ", degrees=True)
    return float(ang[0]), float(ang[1]), float(ang[2])


def write_star(out_path: Path, coords_xyz: np.ndarray, angles: np.ndarray) -> None:
    c = np.asarray(coords_xyz, dtype=np.float32)
    a = np.asarray(angles, dtype=np.float32)
    df = pd.DataFrame(
        {
            "rlnCoordinateX": c[:, 0],
            "rlnCoordinateY": c[:, 1],
            "rlnCoordinateZ": c[:, 2],
            "rlnAngleRot": a[:, 0],
            "rlnAngleTilt": a[:, 1],
            "rlnAnglePsi": a[:, 2],
        }
    )
    starfile.write({"0": df}, out_path, overwrite=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Membrane protein picking: clean membrane -> dilation -> (dilated-clean) CC filter -> convex-normal STAR export.")
    parser.add_argument("--tomo", required=True, help="input tomogram .mrc")
    parser.add_argument("--voxel-size", type=float, required=True, help="voxel size in Angstrom")
    parser.add_argument("--bin", type=int, default=2, help="integer bin factor (default 2)")
    parser.add_argument("--butterworth-cutoff", type=float, default=200.0, help="Butterworth cutoff in Angstrom; <=0 disables (default 200.0)")

    parser.add_argument("--sheetness-percentile", type=float, default=94.0, help="membrane sheetness percentile (default 94.0)")
    parser.add_argument("--min-membrane-voxels", type=int, default=10000, help="minimum membrane CC voxels (default 10000)")
    parser.add_argument("--min-inplane-width", type=float, default=20.0, help="minimum membrane in-plane width in pixels (default 20.0)")
    parser.add_argument("--membrane-border-clear", type=int, default=4, help="clear border voxels in membrane mask (default 4)")

    parser.add_argument("--dilate-radius", type=int, default=2, help="dilation radius in pixels on clean membrane mask (default 2)")
    parser.add_argument("--protein-min-voxels", type=int, default=20, help="minimum candidate CC size in shell (default 20)")
    parser.add_argument("--protein-max-voxels", type=int, default=100, help="maximum candidate CC size in shell (default 100)")

    parser.add_argument("--out-prefix", default=None, help="output prefix path (default: tomo stem in its folder)")
    parser.add_argument("--debug", action="store_true", help="write intermediate masks")
    args = parser.parse_args()

    voxel = float(args.voxel_size)
    if voxel <= 0:
        raise SystemExit("error: --voxel-size must be > 0")

    tomo_path = Path(args.tomo).resolve()
    if not tomo_path.exists():
        raise SystemExit(f"error: tomogram not found: {tomo_path}")

    if args.out_prefix is None:
        out_prefix = tomo_path.parent / tomo_path.stem
    else:
        out_prefix = Path(args.out_prefix)
        if not out_prefix.is_absolute():
            out_prefix = tomo_path.parent / out_prefix

    print(f"loading: {tomo_path}")
    vol = load_mrc(tomo_path)
    print(f"  volume (z,y,x): {vol.shape}")

    bin_factor = max(1, int(args.bin))
    if bin_factor > 1:
        vol_gpu = cp.asarray(vol, dtype=cp.float32)
        vol_gpu = cskimage.transform.downscale_local_mean(vol_gpu, (bin_factor, bin_factor, bin_factor))
        vol = cp.asnumpy(vol_gpu).astype(np.float32)
        voxel *= float(bin_factor)
        print(f"  binned by {bin_factor}, new shape: {vol.shape}, effective voxel: {voxel:.3f} A")
        if args.debug:
            save_mrc(out_prefix.with_name(out_prefix.name + "_0_binned.mrc"), vol, voxel_size=voxel)

    norm = normalize_percentile(vol, low=1.0, high=99.0)
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_1_norm.mrc"), norm, voxel_size=voxel)

    if float(args.butterworth_cutoff) > 0:
        ratio = max(1e-4, min(voxel / float(args.butterworth_cutoff), 0.49))
        filtered = butterworth_preprocess(norm, cutoff_frequency_ratio=ratio, high_pass=False)
        print(f"  Butterworth low-pass: cutoff = {args.butterworth_cutoff} A, ratio = {ratio:.6f}")
    else:
        filtered = norm
        print("  Butterworth disabled")
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_2_butterworth.mrc"), filtered, voxel_size=voxel)

    print("detecting membrane (without dilation)...")
    membrane_raw = detect_membrane_sheetness(filtered, sheetness_percentile=float(args.sheetness_percentile))
    membrane = clean_membrane_mask(membrane_raw, min_voxels=max(1, int(args.min_membrane_voxels)), border_width=max(0, int(args.membrane_border_clear)), min_inplane_width=max(1.0, float(args.min_inplane_width)))
    mem_vox = int(np.sum(membrane > 0.5))
    print(f"  membrane voxels kept: {mem_vox}")
    if mem_vox == 0:
        print("no membrane remains after cleaning; nothing to do")
        return
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_3_membrane_raw.mrc"), membrane_raw, voxel_size=voxel)
        save_mrc(out_prefix.with_name(out_prefix.name + "_4_membrane_clean.mrc"), membrane, voxel_size=voxel)

    print(f"dilating clean membrane mask: radius = {int(args.dilate_radius)} px")
    membrane_dilated = dilate_membrane_mask(membrane, dilate_radius_px=int(args.dilate_radius))
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_5_membrane_dilated.mrc"), membrane_dilated, voxel_size=voxel)

    mem_with_ball = clean_membrane_mask(membrane_raw, min_voxels=max(1, int(args.min_membrane_voxels)), border_width=max(0, int(args.membrane_border_clear)), min_inplane_width=max(1.0, float(args.min_inplane_width)), ball_radius=int(args.dilate_radius))
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_6_membrane_with_ball.mrc"), mem_with_ball, voxel_size=voxel)

    print(f"extracting candidates, keeping CC voxels in [{int(args.protein_min_voxels)}, {int(args.protein_max_voxels)}]")
    protein_mask = extract_potential_membrane_proteins_from_shell(
        membrane_with_ball=mem_with_ball,
        membrane_clean=membrane_dilated,
        min_voxels=max(1, int(args.protein_min_voxels)),
        max_voxels=max(1, int(args.protein_max_voxels)),
    )
    prot_vox = int(np.sum(protein_mask))
    print(f"  candidate membrane-protein voxels: {prot_vox}")
    if prot_vox == 0:
        print("no membrane-bound proteins found; nothing to export")
        return
    if args.debug:
        save_mrc(out_prefix.with_name(out_prefix.name + "_7_protein_mask_filtered.mrc"), protein_mask.astype(np.float32), voxel_size=voxel)

    coords, normals = particles_with_convex_normals(protein_mask=protein_mask, membrane_mask=membrane)
    if len(coords) == 0:
        print("no protein particles converted to centroids; nothing to export")
        return
    print(f"  particles (connected components): {len(coords)}")

    angles = np.zeros((len(coords), 3), dtype=np.float32)
    for i, nvec in enumerate(normals):
        angles[i] = np.array(vector_to_euler_zyz(nvec), dtype=np.float32)

    star_path = out_prefix.with_name(out_prefix.name + "_membrane_proteins.star")
    write_star(star_path, coords, angles)
    print(f"saved STAR: {star_path} ({len(coords)} particles)")


if __name__ == "__main__":
    main()
