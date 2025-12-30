#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import mrcfile
from pathlib import Path
import numpy as np
from tqdm import tqdm


def transform(input: Path, output: Path, bin_factor, overwrite):
    """Use ProcessPool and cv2 to speed up writing pictures."""
    if output.exists() and not overwrite:
        print(f"[INFO] Output exists: {output}. Skipped.")
        return 0
    try:
        mrc = mrcfile.mmap(input, mode='r+')
        img = mrc.data.astype(np.float32) # !!!
        if bin_factor > 1:
            h, w = img.shape[:2]
            new_size = (w // bin_factor, h // bin_factor)
            # INTER_AREA 是下采样（缩放）的最佳插值算法，相当于平均池化
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        if img.max() - img.min() < 1:
            print(f"[INFO] Input {input} all black or white. Skipped.")
            return 0
        # Ultralytics only accept int8 images to be trained and reasoned
        img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        cv2.imwrite(str(output), img_norm)
        mrc.close()
        return 1
    except Exception as e:
        print(f"[ERROR] Failed to process {input}: {e}")
        return 0


def process(in_dir: Path, out_dir: Path, in_ext: str, out_ext: str, bin_factor: int, overwrite: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == in_ext.lower()])
    print(f"[INFO] Found {len(inputs)} micrograph(s).")
    outputs = [out_dir / p.with_suffix(out_ext).name for p in inputs]

    num_workers = min(os.cpu_count(), len(inputs))
    print(f"[INFO] Using {num_workers} process to transform images...")
    success_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(transform, inputs[i], outputs[i], bin_factor, overwrite) for i in range(len(inputs))]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            success_count += f.result()
    print(f"[INFO] Success! Processed {success_count} micrograph(s).")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Transform mrc files into png files.")
    ap.add_argument("-i", "--input", required=True, help="input folder")
    ap.add_argument("-o", "--output", required=True, help="output folder")
    ap.add_argument("--bin", type=int, default=4, help="binning factor for output (default: 4)")
    ap.add_argument("--in-ext", default=".mrc", help="extension for input files (default: .mrc)")
    ap.add_argument("--out-ext", default=".png", help="extension for output files (default: .png)")
    ap.add_argument("--override", action="store_true", help="override existing files")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[ERROR] Path not found: {in_dir}", file=sys.stderr)
        sys.exit(2)
    rc = process(in_dir, out_dir, args.in_ext, args.out_ext, args.bin, args.override)
    sys.exit(rc)


if __name__ == "__main__":
    main()
