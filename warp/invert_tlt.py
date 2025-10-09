#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : invert_tlt.py
# Time       ：2025/10/9 13:39
# Author     ：Jago
# Email      ：huwl@hku.hk
# Description：
Invert the sign of all numeric values in .tlt files.
Examples:
Invert angles in a single folder:
    python invert_tlt_angles.py -i /path/to/tilts -s .bak
Do the same for every subfolder:
    python invert_tlt_angles.py -i /path/to/parent -s .bak --recursive
"""

from pathlib import Path
import argparse
import sys
import shutil


def find_tlt_files(root: Path, recursive: bool):
    """Yield .tlt files whose basename matches the containing folder name."""
    if recursive:
        for p in root.iterdir():
            if p.is_dir():
                folder_name = p.name
                f = p / f"{folder_name}.tlt"
                if f.exists():
                    yield f
    else:
        folder_name = root.name
        f = root / f"{folder_name}.tlt"
        if f.exists():
            yield f


def backup_file(src: Path, suffix: str, overwrite: bool = False) -> Path:
    """Copy src -> src + suffix (same directory). Return backup path."""
    backup = src.with_name(src.name + suffix)
    if backup.exists() and not overwrite:
        print(f"[INFO] Backup exists, skip: {backup}")
        return backup
    shutil.copy2(src, backup)
    return backup


def invert_tlt_values(path: Path, encoding: str = "utf-8"):
    """Read numeric lines, invert sign, and write back."""
    lines = path.read_text(encoding=encoding).splitlines()
    out_lines = []
    for idx, line in enumerate(lines, start=1):
        s = line.strip()
        if not s:
            out_lines.append("\n")
            continue
        try:
            val = float(s)
            out_lines.append(f"{-val}\n")
        except ValueError:
            print(f"[WARN] Non-numeric line {idx}: '{s}'", file=sys.stderr)
            out_lines.append(line + "\n")
    path.write_text("".join(out_lines), encoding=encoding)


def process_folder(root: Path, suffix: str, recursive: bool, overwrite_backup: bool):
    files = list(dict.fromkeys(find_tlt_files(root, recursive)))
    if not files:
        print(f"[INFO] No .tlt files found in {root} (recursive={recursive})")
        return 0, 0

    ok, fail = 0, 0
    for f in files:
        try:
            backup_file(f, suffix, overwrite=overwrite_backup)
            invert_tlt_values(f)
            ok += 1
        except Exception as e:
            print(f"[ERROR] {f}: {e}", file=sys.stderr)
            fail += 1
    return ok, fail


def main():
    ap = argparse.ArgumentParser(description="Invert numeric values in .tlt files (e.g., tilt angles).")
    ap.add_argument("-i", "--input", required=True, help="input folder (operates on this folder or its subfolders when --recursive)")
    ap.add_argument("-s", "--suffix", default=".bak", help="backup suffix to append (default: .bak)")
    ap.add_argument("--recursive", action="store_true", help="process subfolders recursively (default: False)")
    ap.add_argument("--overwrite-backup", action="store_true", help="when creating backups, overwrite existing backup files (default: False)")
    args = ap.parse_args()

    root = Path(args.input)
    if not root.exists() or not root.is_dir():
        print(f"Input {root} not found or not a directory", file=sys.stderr)
        sys.exit(2)

    print("------START------")
    ok, fail = process_folder(root, args.suffix, args.recursive, args.overwrite_backup)
    print(f"------FINISH------\nProcessed: OK={ok}, Failed={fail}")


if __name__ == "__main__":
    main()
