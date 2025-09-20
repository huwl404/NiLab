#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : generate_orderlist.py
# Time       ：2025/9/18 12:01
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
Generate reordered tilt angle lists from .tlt files for dose-symmetric tilt series.
For each folder, the script:
  1. Reads the tilt angles from the <folder>.tlt file.
  2. Validates that the file has exactly --total-row lines.
  3. Reorders rows starting from the specified --zero-row, alternating sides according to --direction, flipping every --flip-after entries.
  4. Writes the reordered list to a CSV file named <folder><suffix> in the same folder.
Examples:
Process a single tilt series folder:
    python generate_orderlist.py -i /path/to/ts_folder
Process multiple folders recursively:
    python generate_orderlist.py -i /path/to/input_dir --recursive
Change flipping frequency and output suffix:
    python generate_orderlist.py -i /path/to/ts_folder --total-row 41 --zero-row 21 --flip-after 1 --suffix _ord
"""
from collections import deque
from pathlib import Path
import argparse
import sys


def build_reordered_indices(n: int, zero_row_idx: int, flip_after: int, direction: str):
    pos_queue = deque(range(zero_row_idx + 1, n + 1))
    neg_queue = deque(range(zero_row_idx - 1, 0, -1))

    result = [zero_row_idx]
    current_side = direction

    while pos_queue or neg_queue:
        take = flip_after
        if current_side == 'pos':
            while take > 0 and pos_queue:
                result.append(pos_queue.popleft())
                take -= 1
            current_side = 'neg'
        else:
            while take > 0 and neg_queue:
                result.append(neg_queue.popleft())
                take -= 1
            current_side = 'pos'

    return result


def process_one_folder(folder: Path, args) -> bool:
    """
    Process one folder:
    - find <foldername>.tlt
    - read lines (strings)
    - validate total-row if provided
    - build reorder indices
    - write CSV named foldername + suffix in the same folder
    Returns True on success (CSV written), False if skipped/failed.
    """
    tlt_path = folder / (folder.name + ".tlt")
    if not tlt_path.exists():
        print(f"[SKIP] {folder}: no {tlt_path.name}")
        return False

    lines = []
    with open(tlt_path, 'r') as tlt:
        for raw in tlt:
            s = raw.strip()
            lines.append(s)

    n = len(lines)
    if n != args.total_row:
        print(f"[SKIP] {folder}: {tlt_path} number of lines {n} != --total-row {args.total_row}")
        return False

    indices = build_reordered_indices(n, args.zero_row, args.flip_after, args.direction)
    output_lines = []
    counter = 1
    for idx in indices:
        # idx is 1-based, map to lines list index 0-based
        val = lines[idx - 1]
        output_lines.append(f"{counter},{val}\n")
        counter += 1

    out_path = folder / (folder.name + args.suffix)
    try:
        with out_path.open('w', newline='') as fh:
            for ln in output_lines:
                fh.write(ln)
    except Exception as e:
        print(f"[SKIP] {folder}: {e}")
        return False

    return True


def main():
    ap = argparse.ArgumentParser(description="Generate order list .csv from .tlt for dose-symmetric tilt series.")
    ap.add_argument("-i", "--input", required=True, help="Input folder (process this folder or its children if --recursive)")
    ap.add_argument("--recursive", action="store_true", help="Process folders recursively (default: False), i.e. the "
                                                             "input folder includes IMOD-processed folders")
    ap.add_argument("--total-row", type=int, default=35, help="Require .tlt to have exactly this many rows (default: 35)")
    ap.add_argument("--zero-row", type=int, default=18, help="Zero angle row (1-based) to start from (default: 18)")
    ap.add_argument("--flip-after", type=int, default=2, help="Take this many items each side before flipping (default 2)")
    ap.add_argument("--direction", choices=("pos", "neg"), default="pos", help="Direction during data collection (e.g. "
                                                                               "pos=0,3,6,-3,-6... neg=0,-3,-6,3,"
                                                                               "6..., default: pos)")
    ap.add_argument("--suffix", default="_test.csv", help="Output CSV filename suffix (default: _test.csv)")
    args = ap.parse_args()

    root = Path(args.input)
    if not root.exists() or not root.is_dir():
        print(f"Input {root} not found or not a directory", file=sys.stderr)
        sys.exit(2)
    if args.zero_row < 1 or args.zero_row > args.total_row:
        print(f"Wrong params: --zero-row {args.zero_row} out of range [1..{args.total_row}]")
        return False

    if args.recursive:
        folders = [p for p in root.iterdir() if p.is_dir()]
    else:
        folders = [root]

    if not folders:
        print(f"{root}: Nothing need to be processed.", file=sys.stderr)
        sys.exit(0)

    ok, fail = 0, 0
    print("------START------")
    for folder in folders:
        if process_one_folder(folder, args):
            print(f"[OK] {folder}")
            ok += 1
        else:
            print(f"[FAILED] {folder}")
            fail += 1

    print(f"Total: {len(folders)}, OK: {ok}, Failed: {fail}")


if __name__ == "__main__":
    main()
