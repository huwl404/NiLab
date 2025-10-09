#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : rename_mdoc.py
# Time       ：2025/9/16 13:18
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
Copy and rename .mdoc files to an output folder.
Behavior:
- If --tsf is provided: try to match extracted digits (with optional prefix/suffix) to a folder inside the tsf folder.
If a matching folder exists, copy the .mdoc and rename to <target_folder_name>.mrc.mdoc (same as before).
- If --tsf is NOT provided: copy **all** .mdoc files to the output folder and rename them.
The new name is constructed from extracted digits when available; prefix and suffix (if provided) are applied to the base name.
Examples:
With tsf: python rename_mdoc.py -m /path/to/mdoc -r /path/to/tiltseries -o outdir --prefix SMV031825_
Without tsf: python rename_mdoc.py -m /path/to/mdoc -o outdir
"""
import argparse
import re
import shutil
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Copy and rename mdoc in original mdoc folder corresponding to referred tilt series folders to "
                    "output folder.")
    p.add_argument("-m", "--mdoc", required=True, help="original mdoc folder including mdoc files")
    p.add_argument("-r", "--tsf", help="reference folder including tilt series folders")
    p.add_argument("-o", "--output", required=True, help="output folder to save the processed mdoc files")
    p.add_argument("--prefix", help="eg. --prefix SMV031725_")
    p.add_argument("--suffix", help="eg. --suffix _SUF")
    p.add_argument("--recursive", action="store_true", help="search mdoc files recursively (default: False)")
    args = p.parse_args()

    mdoc = Path(args.mdoc)
    tsf = Path(args.tsf) if args.tsf else None
    output = Path(args.output)
    ext = ".mdoc"  # only support mdoc so far

    if not mdoc.exists() or not mdoc.is_dir():
        print(f"{mdoc} parsing failed.", file=sys.stderr)
        sys.exit(2)
    if tsf is not None and (not tsf.exists() or not tsf.is_dir()):
        print(f"{tsf} parsing failed.", file=sys.stderr)
        sys.exit(2)
    output.mkdir(parents=True, exist_ok=True)

    if args.recursive:
        files = [p for p in mdoc.rglob(f"*{ext}") if p.is_file()]
    else:
        files = [p for p in mdoc.glob(f"*{ext}") if p.is_file()]

    if not files:
        print(f"{mdoc} contains no mdoc file.")
        return

    total = 0
    copied = 0
    skipped_no_digits = 0
    skipped_no_target = 0

    if tsf is not None:
        for src in files:
            total += 1
            fname = src.name
            digits = re.findall(r'\d+', fname)
            if not digits:
                print(f"Skipped {mdoc}/{fname}: no number extracted from the file name")
                skipped_no_digits += 1
                continue

            nums = ''.join(digits)
            target_folder_name = f"{nums}"
            if args.prefix:
                target_folder_name = f"{args.prefix}{target_folder_name}"
            if args.suffix:
                target_folder_name = f"{target_folder_name}{args.suffix}"

            target_folder = tsf / target_folder_name

            if target_folder.exists() and target_folder.is_dir():
                dest_path = f"{output / target_folder_name}.mrc.mdoc"
                try:
                    shutil.copy2(src, dest_path)
                    # print(f"Matched {mdoc}/{fname} with {target_folder}, Copied and Renamed to {dest_path}")
                    copied += 1
                except Exception as e:
                    print(f"Copy {mdoc}/{fname} to {dest_path} failed: {e}")
            else:
                print(f"Skipped {mdoc}/{fname}: target {target_folder} does not exist")
                skipped_no_target += 1
    else:
        for src in files:
            total += 1
            fname = src.name
            digits = re.findall(r'\d+', fname)
            if not digits:
                print(f"Skipped {mdoc}/{fname}: no number extracted from the file name")
                skipped_no_digits += 1
                continue

            nums = ''.join(digits)
            target_folder_name = f"{nums}"
            if args.prefix:
                target_folder_name = f"{args.prefix}{target_folder_name}"
            if args.suffix:
                target_folder_name = f"{target_folder_name}{args.suffix}"

            dest_path = f"{output / target_folder_name}.mrc.mdoc"
            try:
                shutil.copy2(src, dest_path)
                copied += 1
            except Exception as e:
                print(f"Copy {mdoc}/{fname} to {dest_path} failed: {e}")

    print("------ FINISH ------")
    print(f"Total mdoc files: {total}")
    print(f"Successfully copied: {copied}")
    print(f"Skipped for no matching tilt series folder: {skipped_no_target}")
    print(f"Skipped for not being recognized: {skipped_no_digits}")


if __name__ == '__main__':
    main()
