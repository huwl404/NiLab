#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : flip_xf_tlt.py
# Time       ：2025/10/9 11:17
# Author     ：Jago
# Email      ：huwl@hku.hk
# Description：
Behavior: the script will search the target folder (or its immediate subfolders when --recursive)
    for files ending with .xf and .tlt, copy each to a backup named <orig><suffix>, then reverse the
    textual line order in the original file (write back).
Examples:
Backup and flip files in a single folder:
    python flip_xf_tlt.py -i /path/to/tilts -s .bak
Do the same for every subfolder (recursive):
    python flip_xf_tlt.py -i /path/to/parent -s .bak --recursive
"""

from pathlib import Path
import argparse
import sys
import shutil


def find_target_files(root: Path, recursive: bool):
    """Yield .xf and .tlt files whose basename matches the containing folder name."""
    if recursive:
        for p in root.iterdir():
            if p.is_dir():
                folder_name = p.name
                for ext in (".xf", ".tlt"):
                    f = p / f"{folder_name}{ext}"
                    if f.exists():
                        yield f
    else:
        folder_name = root.name
        for ext in (".xf", ".tlt"):
            f = root / f"{folder_name}{ext}"
            if f.exists():
                yield f


def backup_file(src: Path, suffix: str, overwrite: bool = False) -> Path:
    """Copy src -> src + suffix (same directory). Return backup path."""
    backup = src.with_name(src.name + suffix)
    if backup.exists() and not overwrite:
        # do nothing, keep existing backup
        print(f"[INFO] backup file exists, does not override.")
        return backup
    shutil.copy2(src, backup)
    return backup


def flip_lines_inplace(path: Path, encoding: str = "utf-8"):
    """Read text file, reverse lines, write back preserving newline endings as '\n'."""
    text = path.read_text(encoding=encoding)
    # splitlines(keepends=True) keeps original line endings
    lines = text.splitlines(keepends=True)
    lines.reverse()
    path.write_text("".join(lines), encoding=encoding)


def process_folder(root: Path, suffix: str, recursive: bool, overwrite_backup: bool):
    files = list(dict.fromkeys(find_target_files(root, recursive)))  # de-duplicate
    if not files:
        print(f"[INFO] No .xf/.tlt files found in {root} (recursive={recursive})")
        return 0, 0

    ok = 0
    failed = 0
    for f in files:
        try:
            # make backup then flip
            bak = backup_file(f, suffix, overwrite=overwrite_backup)
            # flip lines in the original file
            flip_lines_inplace(f)
            ok += 1
        except Exception as e:
            print(f"[ERROR] {f}: {e}", file=sys.stderr)
            failed += 1

    return ok, failed


def main():
    ap = argparse.ArgumentParser(description="Backup and reverse line order of .xf and .tlt files.")
    ap.add_argument("-i", "--input", required=True,
                    help="input folder (operates on this folder or its subfolders when --recursive)")
    ap.add_argument("-s", "--suffix", default=".bak", help="backup suffix to append (default: .bak)")
    ap.add_argument("--recursive", action="store_true", help="process subfolders recursively (default: False).")
    ap.add_argument("--overwrite-backup", action="store_true",
                    help="when creating backups, overwrite existing backup files (default: False).")
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
