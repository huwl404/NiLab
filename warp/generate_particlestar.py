#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : generate_particlestar.py
# Time       ：2025/9/23 18:42
# Author     ：Jago
# Email      ：huwl@hku.hk
# Description：
Split a STAR containing many tomograms' particles into per-tomogram STAR files while applying the
'recenter' transformation and a binning factor to particle coordinates.
Mainly referred https://gist.github.com/alisterburt/8744accf3f4696dd6d83fc9c4690612c
For each particle, the script:
  1. origin (rlnOrigin*Angst) [Å] is converted to pixels by dividing by rlnImagePixelSize [Å/pixel]
    and subtracted from rlnCoordinate* [pixels].
  2. a user-provided shift (in Å) is converted to pixels and applied in each particle's *local* coordinate
    system via its rotation matrix (computed from rlnAngleRot/Tilt/Psi).
  3. final coordinates are binned (divide by --bin) and Output per-tomogram star files containing columns:
    _rlnCoordinateX _rlnCoordinateY _rlnCoordinateZ _rlnAngleRot _rlnAngleTilt _rlnAnglePsi _rlnMicrographName
Examples:
    python generate_particlestar.py -i relion/Refine3D/job051/run_data.star -o particles -b 4
"""
import argparse
import sys
from pathlib import Path
from typing import Tuple
import einops
import numpy as np
import starfile
from scipy.spatial.transform import Rotation as R


def shift_then_rotate_particles(
    particle_positions: np.ndarray,        # (n,3)
    particle_orientations: np.ndarray,    # (n,3,3)
    shift_pixels: np.ndarray,             # either (3,) or (n,3) in pixels -- local frame
    rotation: np.ndarray = None           # (3,3) global rotation to apply to orientations (optional)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply local shifts (in pixels) to particle positions. shift_pixels can be:
      - shape (3,) : same local shift for all particles
    Returns: (updated_positions (n,3), updated_orientations (n,3,3))
    """
    shift_col = einops.rearrange(shift_pixels, 'xyz -> xyz 1')  # (3,1)
    local_shifts = particle_orientations @ shift_col
    local_shifts = einops.rearrange(local_shifts, 'b xyz 1 -> b xyz')

    updated_particle_positions = particle_positions + local_shifts
    updated_orientations = particle_orientations @ rotation

    return updated_particle_positions, updated_orientations


def process_star(in_star: Path, output_dir: Path, bin_factor: int, shift_pixels: Tuple[float, float, float]):
    """
    Main worker: read star, recenter & shift, bin coordinates and write per-tomo star files.
    """
    star = starfile.read(in_star, always_dict=True)
    if 'particles' not in star or 'optics' not in star:
        raise ValueError("Input STAR must contain 'particles' and 'optics' blocks (RELION 3.1+ style).")

    particles = star['particles']
    optics = star['optics']

    # merge optics into particles to have per-particle pixel size etc.
    df = particles.merge(optics, on='rlnOpticsGroup')
    n = len(df)
    if n == 0:
        raise ValueError("No particles found in STAR file.")

    # extract arrays
    coords = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy(dtype=float)
    pixel_size = df['rlnImagePixelSize'].to_numpy(dtype=float)  # Å per pixel, shape (n,)
    eulers = df[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy(dtype=float)  # degrees (n,3)

    # origins: if missing, use zeros per particle
    if all(col in df.columns for col in ('rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst')):
        origins_ang = df[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']].to_numpy(dtype=float)  # Å
    else:
        origins_ang = np.zeros((n, 3), dtype=float)

    # convert origins (Å) -> pixels: divide by pixel_size (Å/pixel)
    pixel_size_col = einops.rearrange(pixel_size, 'b -> b 1')  # (n,1)
    origins_pix = origins_ang / pixel_size_col  # (n,3) in pixels

    # apply origin correction: coords are in pixels -> subtract origins_pix
    coords_corrected = coords - origins_pix  # (n,3)
    # build rotation matrices: as before, use ZYZ euler, invert to get particle->global mapping
    rot_mats = R.from_euler(angles=eulers, seq='ZYZ', degrees=True).inv().as_matrix()  # (n,3,3)
    shift_pixels = np.asarray(shift_pixels, dtype=float)
    new_coords, _ = shift_then_rotate_particles(
        particle_positions=coords_corrected,
        particle_orientations=rot_mats,
        shift_pixels=shift_pixels,
        rotation=np.eye(3)
    )

    binned_coords = new_coords / float(bin_factor)

    # gather per-tomogram groups
    tomo_names = df['rlnTomoName'].to_numpy()
    unique_tomos = np.unique(tomo_names)

    for tomo in unique_tomos:
        mask = (tomo_names == tomo)
        tomo_coords = binned_coords[mask]  # (m,3)
        tomo_eulers = eulers[mask]         # (m,3)

        out_path = output_dir / f"{tomo}.star"
        with out_path.open("w", newline="") as fh:
            fh.write("data_\n\nloop_\n")
            fh.write("_rlnCoordinateX #1\n")
            fh.write("_rlnCoordinateY #2\n")
            fh.write("_rlnCoordinateZ #3\n")
            fh.write("_rlnAngleRot #4\n")
            fh.write("_rlnAngleTilt #5\n")
            fh.write("_rlnAnglePsi #6\n")
            fh.write("_rlnMicrographName #7\n")

            for (x, y, z), (rot, tilt, psi) in zip(tomo_coords, tomo_eulers):
                fh.write(f" {x:.8f} {y:.8f} {z:.8f} {rot:.6f} {tilt:.6f} {psi:.6f} {tomo}.tomostar\n")

    return len(unique_tomos)


def main():
    ap = argparse.ArgumentParser(description="Split particles from a multi-tomogram STAR into per-tomo STARs after "
                                             "applying recenter and local shift, then bin coordinates.")
    ap.add_argument("-i", "--input", required=True, help="input STAR file (must contain particles and optics)")
    ap.add_argument("-o", "--output", required=True, help="output directory for per-tomo star files")
    ap.add_argument("-b", "--bin", type=int, default=1, help="bin factor to apply to coordinates (default 1)")
    ap.add_argument("--shift", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                    help="local shift in pixels (X Y Z, default=0.0, 0.0, 0.0), will be applied in coordinates")
    args = ap.parse_args()

    in_star = Path(args.input)
    out_dir = Path(args.output)
    if not in_star.exists() or not in_star.is_file():
        print(f"Input {in_star} not found or not a file.", file=sys.stderr)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("------START------")
    n_tomos = process_star(in_star, out_dir, args.bin, tuple(args.shift))

    print(f"Written {n_tomos} per-tomo STAR files to {out_dir}/")


if __name__ == "__main__":
    main()
