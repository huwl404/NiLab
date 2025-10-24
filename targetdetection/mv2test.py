#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : mv2test.py
# Time       : 2025/10/21 14:06
# Author     : Jago
# Email      : huwl@hku.hk
# Description：
"""
import argparse
import sys
import shutil
from pathlib import Path
import random


def find_pairs(map_dir: Path, label_dir: Path, map_ext: str, label_ext: str):
    """Return sorted list of (map_path, label_path) pairs where basenames match."""
    maps = sorted([p for p in map_dir.iterdir() if p.is_file() and p.suffix.lower() == map_ext.lower()])
    pairs = []
    for m in maps:
        base = m.stem
        lab = label_dir / (base + label_ext)
        if lab.exists() and lab.is_file():
            pairs.append((m, lab))
    return pairs


def move_pair(map_src: Path, label_src: Path, map_dst_dir: Path, label_dst_dir: Path, overwrite: bool):
    map_dst = map_dst_dir / map_src.name
    label_dst = label_dst_dir / label_src.name

    def _move(src, dst):
        if dst.exists():
            if overwrite:
                try:
                    dst.unlink()
                except Exception as e:
                    print(f"[WARN] Cannot remove existing file {dst}: {e}")
            else:
                print(f"[INFO] Destination exists, skipping: {dst}")
                return False
        try:
            shutil.move(str(src), str(dst))
            return True
        except Exception as e:
            print(f"[ERROR] Moving {src} -> {dst} failed: {e}")
            return False

    ok_map = _move(map_src, map_dst)
    ok_label = _move(label_src, label_dst)
    return ok_map, ok_label


def main():
    ap = argparse.ArgumentParser(description="Move random matching map/label pairs to an output folder.")
    ap.add_argument("--maps", "-m", required=True, help="folder with image files")
    ap.add_argument("--labels", "-l", required=True, help="folder with label .txt files (same basename)")
    ap.add_argument("--map-ext", default=".png", help="extension for map files (default: .png)")
    ap.add_argument("--txt-ext", default=".txt", help="extension for label files (default: .txt)")
    ap.add_argument("-o", "--out", default="selected", help="output folder (will contain map/ and label/)")
    ap.add_argument("-c", "--count", type=int, default=16, help="number of pairs to move (default: 16)")
    ap.add_argument("--override", action="store_true", help="overwrite files in destination if exist")
    args = ap.parse_args()

    map_dir = Path(args.maps)
    label_dir = Path(args.labels)
    if not map_dir.exists() or not map_dir.is_dir():
        print(f"[ERROR] maps folder not found: {map_dir}", file=sys.stderr)
        sys.exit(2)
    if not label_dir.exists() or not label_dir.is_dir():
        print(f"[ERROR] labels folder not found: {label_dir}", file=sys.stderr)
        sys.exit(2)

    pairs = find_pairs(map_dir, label_dir, args.map_ext, args.txt_ext)
    if not pairs:
        print("[ERROR] No matching pairs found. Ensure basenames match between maps and labels.", file=sys.stderr)
        sys.exit(1)

    total_pairs = len(pairs)
    print(f"[INFO] Found {total_pairs} matching pairs.")

    chosen = pairs.copy()
    random.shuffle(chosen)
    chosen = chosen[: args.count]

    print(f"[INFO] Will move {len(chosen)} pair(s) to {args.out}.")

    out_dir = Path(args.out)
    map_out = out_dir / "map"
    label_out = out_dir / "label"
    out_dir.mkdir(parents=True, exist_ok=True)
    map_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    moved = 0
    failed = 0
    for map_p, label_p in chosen:
        ok_map, ok_label = move_pair(map_p, label_p, map_out, label_out, args.override)
        if ok_map and ok_label:
            moved += 1
            print(f"[OK] Moved: {map_p.name}, {label_p.name}")
        else:
            failed += 1
            print(f"[FAILED] Pair move incomplete: {map_p.name}, {label_p.name}")

    print("------FINISH------")
    print(f"Total found: {total_pairs}, Requested moved: {len(chosen)}, Successfully moved: {moved}, Failed: {failed}")
    return 0


if __name__ == "__main__":
    main()
