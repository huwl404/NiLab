#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : lstFiltered2star.py
# Time       ：2025/7/12 12:29
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
This script filters a RELION .star file using a Jalign .lst output based on a specified parameter.
It selects particles from the STAR file whose corresponding values in the .lst file meet a threshold condition
(greater than or less than), and writes a new filtered .star file retaining the original optics and metadata structure.
Features:
Supports filtering based on any numeric field in the .lst file (e.g., score, zScore, defocus, etc.)
Option to apply absolute value transformation before threshold comparison
Retains all optics groups and STAR metadata structure
Automatically maps particle indices from EMAN2 .lst format to RELION-compatible image names
Usage Example:
    python lstFiltered2star.py --lst1 jalign_output.lst --lst2 jalign_input.lst --star subset1_particles.star \
        --out filtered.star --param score --threshold 0.04 --abs --mode gt
"""
import argparse
import sys
import numpy as np
from EMAN2star import StarFile3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lst1', required=True, help="Jalign output .lst file")
    p.add_argument('--lst2', required=True, help="Jalign input .lst file")
    p.add_argument('--star', required=True, help="Relion one subset .star file")
    p.add_argument('--out', required=True, help="Filtered .star output file")
    p.add_argument('--param', required=True, help="Parameter name to filter on lst1")
    p.add_argument('--threshold', type=float, required=True, help="Threshold for filtering")
    p.add_argument('--abs', action='store_true', help="Apply absolute value before comparison")
    p.add_argument('--mode', choices=['gt', 'lt'], default='gt',
                   help="Keep values greater than (gt) or less than (lt) threshold")
    return p.parse_args()


def read_lst1(path, param, threshold, mode, use_abs):
    """extract indices where param compares to threshold. Returns a set of zero-based indices."""
    keep = set()  # disorder
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            # find param=value field
            val = None
            for field in parts[1:]:
                if field.startswith(f"{param}="):
                    try:
                        v = float(field.split('=', 1)[1])
                        val = abs(v) if use_abs else v
                    except ValueError:
                        sys.exit(f"ERROR: cannot parse value for {param} on line: {line}")
                    break
            if val is None:
                continue  # parameter not found
            if (mode == 'gt' and val > threshold) or (mode == 'lt' and val < threshold):
                keep.add(idx)
    return keep


def read_lst2(path):
    """row index->rlnImageName for kept indices."""
    mapping = {}
    row_id = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            try:
                image_id = int(parts[0])
            except ValueError:
                continue
            micro_path = parts[1]
            # 将 EMAN2 使用的图像编号（从 0 开始）转换为 RELION 中的图像编号（从 1 开始）!
            new_image_id = image_id + 1
            # 前缀 0 补齐
            tag = f"{str(new_image_id).zfill(6)}@{micro_path}"
            # 两个 lst 文件行号是对应的，此处行号并非首列的数字，literally行号
            mapping[row_id] = tag
            row_id = row_id + 1
    return mapping


def filter_star(in_star, out_star, keep_images):
    """filter data_particles by rlnImageName in keep_images set, write new star preserving optics and other blocks."""
    star = StarFile3(in_star)
    # Preserve all blocks
    output_blocks = {}
    for block_name, block in star.items():
        if 'rlnImageName' in block:
            image_names = star[block_name]['rlnImageName']
            if not isinstance(image_names, np.ndarray):
                image_names = np.array(image_names)
            image_names = image_names.astype(str)
            mask = np.isin(image_names, list(keep_images))

            kept_indices = np.nonzero(mask)[0]
            if len(kept_indices) == 0:
                continue

            filtered_block = {
                k: (np.array(v)[kept_indices].tolist() if isinstance(v, (list, np.ndarray)) else v)
                for k, v in block.items()
            }
            output_blocks[block_name] = filtered_block
        else:
            output_blocks[block_name] = block

    # Write all blocks
    with open(out_star, 'w') as f:
        f.write("# version 30001\n\n")
        for block_name, block in output_blocks.items():
            f.write(f"data_{block_name}\n\n")
            if not block:
                continue
            f.write("loop_\n")
            keys = list(block.keys())
            for i, key in enumerate(keys):
                f.write(f"_{key} #{i + 1}\n")
            rows = zip(*(block[k] for k in keys))
            for row in rows:
                f.write(" ".join(str(x) for x in row) + "\n")
            f.write("\n")


def main():
    args = parse_args()

    keep_idxs = read_lst1(args.lst1, args.param, args.threshold, args.mode, args.abs)
    print(f"Kept {len(keep_idxs)} particles in {args.lst1}.")
    if not keep_idxs:
        sys.exit("No entries pass the filter; output would be empty.")

    mapping = read_lst2(args.lst2)
    # Build set of image tags to keep in STAR
    keep_images = {mapping[idx] for idx in mapping if idx in keep_idxs}
    print(f"Kept {len(keep_images)} particles in {args.lst2}.")
    if not keep_images:
        sys.exit("Filtered indices not found in mapping file.")

    filter_star(args.star, args.out, keep_images)
    print(f"Filtered STAR written to {args.out}, kept {len(keep_images)} particles.")


if __name__ == '__main__':
    main()
