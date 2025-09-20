#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : split_tiltstack.py
# Time       ：2025/9/16 14:31
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
Split IMOD-processed tilt series stacks into individual MRC images using .tlt angle files.
For each folder, the script:
  1. Reads the tilt angles from the .tlt file.
  2. Splits the tilt series stack (.mrc) into individual images.
  3. Saves each image to the output folder with filenames of the form <folder>_<angle>.mrc.
  4. Logs successfully processed folders into a log file, so they are not re-processed.
Examples:
Process one IMOD-processed tilt series folder:
    python split_tiltstack.py -i /path/to/ts_folder -o outdir
Process multiple IMOD-processed folders under an input directory:
    python split_tiltstack.py -i /path/to/input_dir -o outdir --recursive
Use 8 workers in parallel and custom log file:
    python split_tiltstack.py -i /path/to/input_dir -o outdir --workers 8 --log mylog.txt --recursive
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import mrcfile


def process_one_folder(folder: Path, output_folder: Path):
    stack_name = folder.name + ".mrc"
    tlt_name = folder.name + ".tlt"
    stack_path, tlt_path = folder / stack_name, folder / tlt_name
    if stack_path is None:
        print(f"[SKIP] {folder}: No {stack_path} file.")
        return False

    if tlt_path is None:
        print(f"[SKIP] {folder}: No {tlt_path} file.")
        return False

    rows = []
    with open(tlt_path, 'r') as tlt:
        for r in tlt:
            if not r:
                continue
            angle_str = r.strip()
            angle = float(angle_str) if angle_str != '' else None
            rows.append(angle)

    if not rows:
        print(f"[SKIP] {folder}: No rows in {tlt_path} file.")
        return False

    with mrcfile.open(stack_path, permissive=True) as ts:  # permissive to tolerate some stack missing part of data
        if ts.data is None:
            print(f"[SKIP] {folder}: {stack_path} has no valid data block.")
            return False

        if len(rows) != len(ts.data):
            print(f"[SKIP] {folder}: Not matching sections between {stack_path} and {tlt_path}.")
            return False

        try:
            for angle, image in zip(rows, ts.data):
                output_name = folder.name + "_" + str(angle) + ".mrc"
                output_path = output_folder / output_name
                mrcfile.write(output_path, image)
        except ValueError as e:
            print(f"[SKIP] {folder}: {e}.")
            return False

    return True


def main():
    ap = argparse.ArgumentParser(description="Split IMOD-processed tilt series stacks into individual MRC images "
                                             "using .tlt angle files.")
    ap.add_argument("-i", "--input", required=True, help="input folder containing multi IMOD-processed folders")
    ap.add_argument("--recursive", action="store_true", help="process folders recursively (default: False), i.e. the "
                                                             "input folder includes IMOD-processed folders")
    ap.add_argument("-o", "--output", default="./frames", help="output folder (default: ./frames)")
    ap.add_argument("--log", default="./processed_ts.log", help="log file containing processed folders (default: "
                                                                "./processed_ts.log)")
    ap.add_argument("--workers", type=int, default=4, help="parallel workers (default: 4)")

    args = ap.parse_args()

    input_folder = Path(args.input)
    log_file = Path(args.log)
    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Input {input_folder} not found or not a directory", file=sys.stderr)
        sys.exit(2)
    if not log_file.exists() or not log_file.is_file():
        print(f"Log {log_file} not found or not a file", file=sys.stderr)
        sys.exit(2)

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    processed = set()
    with open(args.log, "r") as f:
        for line in f:
            if line.strip() and line.strip().startswith(input_folder.name):
                processed.add(line.strip().split('/')[-1])

    if args.recursive:
        # p is PATH, will never be equal with string
        folders = [p for p in input_folder.iterdir() if p.is_dir() and p.name not in processed]
    else:
        folders = [input_folder]

    if not folders:
        print(f"{input_folder}: Nothing need to be processed.")
        sys.exit(0)

    ok, fail = 0, 0
    print("------START------")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one_folder, folder, output_folder): folder for folder in folders}
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
                    with open(args.log, "a") as logf:
                        logf.write(str(folder) + "\n")
                else:
                    print(f"[FAILED] {folder}")
                    fail += 1

    print(f"Total: {len(folders)}, OK: {ok}, Failed: {fail}")


if __name__ == "__main__":
    main()
