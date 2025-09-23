#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : generate_tomostar.py
# Time       ：2025/9/16 21:54
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
Generate .tomostar files for Warp from IMOD-processed tilt series using order CSV, .tlt, and frame averages.
For each tilt-series folder, the script:
  1. Reads the order CSV (<folder>_test.csv by default) containing tilt order and angles.
  2. Builds the expected dose-symmetric tilt sequence (using --total-row, --increase, --flip-after, --direction).
  3. Matches angles to doses and locates corresponding frame-average MRC files from --frame-dir.
  4. Computes the median intensity for each tilt by subsampling pixels (default every 10).
  5. Writes a .tomostar file with columns: wrpMovieName, wrpAngleTilt, wrpAxisAngle, wrpDose, wrpAverageIntensity, wrpMaskedFraction.
Examples:
Process one tilt series folder:
    python generate_tomostar.py -i /path/to/ts_folder -f /path/to/frames
Process multiple folders under a root directory:
    python generate_tomostar.py -i /path/to/input_dir -f /path/to/frames --recursive --workers 16
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv
import sys
from typing import List
import numpy as np
import mrcfile


def compute_tilt_median_intensities(path: Path, sample_factor: int = 10) -> float:
    """
    For each tilt: Flatten the image into 1D, subsample at a fixed interval (default: every 10 pixels),
    then take the median of the subsampled samples as the AverageIntensity for that tilt. simpler implementation of
    https://github.com/warpem/warp/blob/main/WarpTools/Commands/Tiltseries/ImportTiltseries.cs
    """
    if sample_factor < 1:
        raise ValueError("sample_factor must be >= 1")

    try:
        with mrcfile.open(path, permissive=True) as mrc:
            data = mrc.data  # lazy array-like
            if data is None:
                return float("nan")

            arr = np.asarray(data)
            flat = arr.ravel().astype(np.float16)

            finite_mask = np.isfinite(flat)
            if not np.any(finite_mask):
                return float("nan")

            finite_vals = flat[finite_mask]
            n = finite_vals.size

            # 计算最大可用长度（整除 sample_factor）
            usable_len = (n // sample_factor) * sample_factor
            if usable_len == 0:
                # not enough pixels to sample at given factor
                sample_vals = finite_vals
            else:
                # 等间隔选择：0, sample_factor, 2*sample_factor, ...
                idxs = np.arange(0, usable_len, sample_factor)
                sample_vals = finite_vals[idxs]

            median_val = float(np.median(sample_vals))
            return median_val
    except Exception as e:
        print(f"[WRONG] Failed to read '{path}': {e}", file=sys.stderr)
        return float("nan")


def build_angle_sequence(total_row: int, increase: int, flip_after: int, direction: str) -> List[float]:
    """
    Build angle list length total_row.
    Pattern: offsets sequence = [0, +1,+2, -1,-2, +3,+4, -3,-4, ...]  (flip_after controls group size)
    angle = offset * increase
    'direction' controls which sign-group goes first: 'pos' -> + then - ; 'neg' -> - then +.
    """
    if flip_after <= 0:
        raise ValueError("flip-after must be > 0")
    if direction not in ("pos", "neg"):
        raise ValueError("direction must be 'pos' or 'neg'")

    needed = total_row - 1
    offsets = [0]
    k = 1
    # how many entries would a full block (pos + neg) need?
    full_block_len = 2 * flip_after

    while needed > 0:
        block = list(range(k, k + flip_after))
        if needed >= full_block_len:
            # can append full block (pos then neg or neg then pos)
            if direction == "pos":
                offsets.extend(block)
                offsets.extend([-x for x in block])
            else:
                offsets.extend([-x for x in block])
                offsets.extend(block)
            needed -= full_block_len
        else:
            if needed % 2 == 0:
                half = needed // 2
                if direction == "pos":
                    offsets.extend(block[:half])
                    offsets.extend([-x for x in block[:half]])
                else:
                    offsets.extend([-x for x in block[:half]])
                    offsets.extend(block[:half])
                needed = 0
            else:
                raise ValueError("total-row must be odd")
        k += flip_after

    angles = [int(o * increase) for o in offsets]
    return angles


def process_one_folder(folder: Path, frame_dir: Path, output_dir: Path, args):
    folder_name = folder.name
    csv_path = folder / (folder_name + args.csv_suffix)
    if not csv_path.exists():
        print(f"[SKIP] {folder}: {csv_path.name} not found")
        return False

    order_rows = []
    with open(csv_path, "r", newline='') as fh:
        rdr = csv.reader(fh)
        for _, r in enumerate(rdr):
            order = int(r[0].strip())
            angle = float(r[1].strip())
            order_rows.append((order, angle))

    if not order_rows:
        print(f"[SKIP] {folder}: No rows in {csv_path} file.")
        return False

    angle_seq = build_angle_sequence(args.total_row, args.increase, args.flip_after, args.direction)
    dose_seq = [t * args.exposure for t in range(0, args.total_row)]

    star_rows = []
    zero = order_rows[0][1]
    name_len = len(os.path.relpath(Path(frame_dir).resolve(), Path(output_dir).resolve())) + 16
    fmt_dec = args.fmt_decimals
    dec_len = fmt_dec + 8
    for (order, angle) in order_rows:
        if len(order_rows) == args.total_row:
            dose = dose_seq[order - 1]
        else:
            rel = round(angle - zero)
            index = angle_seq.index(rel)
            dose = dose_seq[index]

        frame_path = frame_dir / (folder_name + "_" + str(angle) + ".mrc")
        if not frame_path.exists():
            print(f"[SKIP] {folder}: {frame_path} not found")
            return False

        wrp_fs = os.path.relpath(Path(args.warp_frameseries).resolve(), Path(output_dir).resolve())
        wrpMovieName = wrp_fs + "/" + frame_path.name
        avg_int = compute_tilt_median_intensities(frame_path, sample_factor=args.sample_factor)
        axis_angle = args.axis_angle
        masked_fraction = args.masked_fraction

        star_rows.append({
            "wrpMovieName": wrpMovieName,
            "wrpAngleTilt": angle,
            "wrpAxisAngle": axis_angle,
            "wrpDose": dose,
            "wrpAverageIntensity": avg_int,
            "wrpMaskedFraction": masked_fraction
        })

    out_star = output_dir / (folder_name + ".tomostar")
    # sort on angle
    star_rows.sort(key=lambda r: r["wrpAngleTilt"])
    with out_star.open("w", newline="") as fh:
        fh.write("data_\n\nloop_\n")
        fh.write(f"_wrpMovieName #1\n")
        fh.write(f"_wrpAngleTilt #2\n")
        fh.write(f"_wrpAxisAngle #3\n")
        fh.write(f"_wrpDose #4\n")
        fh.write(f"_wrpAverageIntensity #5\n")
        fh.write(f"_wrpMaskedFraction #6\n")

        for r in star_rows:
            # format numbers
            ang_s = f"{r['wrpAngleTilt']:{dec_len}.{fmt_dec}f}"
            axis_s = f"{r['wrpAxisAngle']:{dec_len}.{fmt_dec}f}"
            dose_s = f"{r['wrpDose']:{dec_len}.{fmt_dec}f}"
            avg_s = f"{r['wrpAverageIntensity']:{dec_len}.{fmt_dec}f}"
            mask_s = f"{r['wrpMaskedFraction']:{dec_len}.{fmt_dec}f}"
            movie = r['wrpMovieName'].ljust(name_len)
            fh.write(f"{movie}{ang_s}{axis_s}{dose_s}{avg_s}{mask_s}\n")

    return True


def main():
    ap = argparse.ArgumentParser(description="Generate .tomostar files for Warp from IMOD-processed tilt series using "
                                             "order CSV, .tlt, and frame averages. ")
    ap.add_argument("-i", "--input", required=True, help="tilt-series root folder if --recursive or single folder")
    ap.add_argument("-f", "--frame-dir", required=True, help="frame-series-average root folder (contains "
                                                             "<folder>_<angle>.mrc files)")
    ap.add_argument("--recursive", action="store_true", help="process folders recursively (default: False), i.e. the "
                                                             "input folder includes IMOD-processed folders")
    ap.add_argument("-o", "--output", default="./tomostar", help="output directory for .star files (default ./tomostar)")
    ap.add_argument("--workers", type=int, default=4, help="parallel workers (default 4)")
    ap.add_argument("--csv-suffix", default="_test.csv", help="csv suffix (default _test.csv)")
    ap.add_argument("--total-row", type=int, default=35, help="expected total rows (default 35)")
    ap.add_argument("--exposure", type=float, default=3.0, help="exposure for each tilt (default 3)")
    ap.add_argument("--increase", type=float, default=3.0, help="tilt increase step (default 3)")
    ap.add_argument("--flip-after", type=int, default=2, help="take this many items each side before flipping ("
                                                              "default 2)")
    ap.add_argument("--direction", choices=("pos", "neg"), default="pos", help="Direction during data collection (e.g. "
                                                                               "pos=0,3,6,-3,-6... neg=0,-3,-6,3,"
                                                                               "6..., default: pos)")
    ap.add_argument("--axis-angle", type=float, default=-94.0, help="_wrpAxisAngle value (default -94.0)")
    ap.add_argument("--masked-fraction", type=float, default=0.0, help="_wrpMaskedFraction value (default 0.0)")
    ap.add_argument("--sample-factor", type=int, default=10, help="sampling factor for _wrpAverageIntensity "
                                                                  "computation (default 10, same as warp source code)")
    ap.add_argument("--warp-frameseries", default="./warp_frameseries",
                    help="the folder name after doing fs_motion_and_ctf or fs_ctf will be written in output .tomostar file "
                         "(default: ./warp_frameseries, dont ask me why, I dont know why Warp points to non-existing files.)")
    ap.add_argument("--fmt-decimals", type=int, default=2, help="decimal places in star output (default 2)")
    args = ap.parse_args()

    root = Path(args.input)
    frame_dir = Path(args.frame_dir)
    if not root.exists() or not root.is_dir():
        print(f"Input {root} not found or not a directory", file=sys.stderr)
        sys.exit(2)
    if not frame_dir.exists() or not frame_dir.is_dir():
        print(f"Frame dir {frame_dir} not found or not a directory", file=sys.stderr)
        sys.exit(2)
    if args.total_row % 2 == 0:
        print(f"Wrong {args.total_row}: This script can only handle dose-symmetric data", file=sys.stderr)
        sys.exit(2)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.recursive:
        folders = [p for p in root.iterdir() if p.is_dir()]
    else:
        folders = [root]

    if not folders:
        print(f"{root}: Nothing need to be processed.", file=sys.stderr)
        sys.exit(0)

    ok, fail = 0, 0
    print("------START------")
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(process_one_folder, folder, frame_dir, output_dir, args): folder for folder in folders}
        for fut in as_completed(futures):
            folder = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[ERROR] {folder}: worker raised exception: {e}", file=sys.stderr)
                fail += 1
            else:
                if res:
                    print(f"[OK] {folder}")
                    ok += 1
                else:
                    print(f"[FAILED] {folder}")
                    fail += 1

    print(f"Total: {len(folders)}, OK: {ok}, Failed: {fail}")


if __name__ == "__main__":
    main()
