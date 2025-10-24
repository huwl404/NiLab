#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : batch_runner.py
# Time       : 2025/10/24 9:17
# Author     : Jago
# Email      : huwl@hku.hk
# Description：
"""
import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def chunk_list(items: List[Path], max_per_chunk: int) -> List[List[Path]]:
    if max_per_chunk <= 0:
        raise ValueError("max_per_chunk must be > 0")
    chunks = []
    for i in range(0, len(items), max_per_chunk):
        # slice won't trigger Out of Index error
        chunks.append(items[i:i+max_per_chunk])
    return chunks


def run_command(cmd: List[str], cwd: Path = None) -> Tuple[int, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
        output = proc.stdout if proc.stdout is not None else ''
        return proc.returncode, output
    except Exception as e:
        return 1, f"Exception while running command {cmd}: {e}"


def main():
    ap = argparse.ArgumentParser(description="Batch runner for WarpTools create_settings + fs_ctf over chunks of files")
    ap.add_argument('--folder_data', required=True, help='folder with image files (e.g. frames)')
    ap.add_argument('--folder_processing', required=True, help='base folder for processing outputs (e.g. warp_frameseries)')
    ap.add_argument('--output', required=True, help='base output settings filename (e.g. warp_frameseries.settings)')

    ap.add_argument('--extension', default='*.mrc', help='image files\' extension (default: "*.mrc")')
    ap.add_argument('--angpix', default=1.571, type=float, help='create_settings parameter (default: 1.571)')
    ap.add_argument('--exposure', default=3, type=float, help='create_settings parameter (default: 3)')
    ap.add_argument('--grid', default='2x2x1', help='fs_ctf parameter (default: 2x2x1)')
    ap.add_argument('--range_max', default=7, type=float, help='fs_ctf parameter (default: 7)')
    ap.add_argument('--defocus_max', default=8, type=float, help='fs_ctf parameter (default: 8)')
    ap.add_argument('--dont_use_sum', action='store_true', help='fs_ctf parameter (default: use_sum); if you assign this,'
                                                                ' use_sum won\'t be used for fs_ctf')
    ap.add_argument('--perdevice', default=4, type=int, help='fs_ctf parameter (default: 4)')

    ap.add_argument('--max_files', default=20000, type=int, help='max files per chunk (default: 20000)')
    ap.add_argument('--log', default='batch_runner.log', help='log file (default: batch_runner.log)')

    args = ap.parse_args()

    folder_data = Path(args.folder_data)
    if not folder_data.exists() or not folder_data.is_dir():
        print(f"[Error] {folder_data} not found or not a directory.")
        sys.exit(2)

    # collect files (non-recursive)
    files = sorted(folder_data.glob(args.extension))
    n = len(files)
    if n == 0:
        print(f"[Error] No files found in {folder_data}.")
        sys.exit(2)
    print(f"[INFO] Found {n} files matching pattern '{args.extension}' in {folder_data}")

    max_per = args.max_files
    chunks = chunk_list(files, max_per)
    x = len(chunks)
    print(f"[INFO] Splitting into {x} chunk(s) (<= max {max_per} files each)")

    failed_chunks = []
    for i, chunk_files in enumerate(chunks):
        prefix = f"{i}_"
        # create chunk folder name: same parent as folder_data
        chunk_folder = folder_data.with_name(prefix + folder_data.name)
        if chunk_folder.exists() and chunk_folder.is_dir():
            print(f"[INFO] Removing existing directory {chunk_folder} (won\'t affect original frames)")
            # delete a folder containing symlinks won't affect the original data!!!
            shutil.rmtree(chunk_folder)
        chunk_folder.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Creating symlinks for chunk {i}: {len(chunk_files)} files -> {chunk_folder}")
        for src in chunk_files:
            dst = chunk_folder / src.name
            try:
                os.symlink(src.resolve(), dst)
            except Exception as e:
                print(f"[WARN] failed to link {src} -> {dst}: {e}")

        proc_chunk = prefix + args.folder_processing
        out_chunk = prefix + args.output

        # Run WarpTools create_settings
        cmd1 = [
            'WarpTools', 'create_settings',
            '--folder_data', str(chunk_folder),
            '--folder_processing', str(proc_chunk),
            '--output', str(out_chunk),
            '--extension', args.extension,
            '--angpix', str(args.angpix),
            '--exposure', str(args.exposure)
        ]

        print(f"[INFO] Running create_settings for chunk {i}:")
        print(" ".join(cmd1))
        rc1, out1 = run_command(cmd1)
        with open(args.log, 'a', encoding='utf-8') as f:
            f.write(datetime.now().isoformat(sep=' ', timespec='seconds') + "\n")
            f.write(out1 + "\n")

        if rc1 != 0 or ('Exception' in out1):
            print(f"[FAILED] create_settings FAILED for chunk {i} (rc={rc1}); see {args.log}")
            failed_chunks.append(i)
            continue

        # Run WarpTools fs_ctf
        cmd2 = [
            'WarpTools', 'fs_ctf',
            '--settings', str(out_chunk),
            '--grid', args.grid,
            '--range_max', str(args.range_max),
            '--defocus_max', str(args.defocus_max),
            '--perdevice', str(args.perdevice),
            '--use_sum'
        ]
        if args.dont_use_sum:
            cmd2 = cmd2[:-1]

        print(f"[INFO] Running fs_ctf for chunk {i}:")
        print(" ".join(cmd2))
        rc2, out2 = run_command(cmd2)
        with open(args.log, 'a', encoding='utf-8') as f:
            f.write(datetime.now().isoformat(sep=' ', timespec='seconds') + "\n")
            f.write(out2 + "\n")

        if rc2 != 0 or ('Exception' in out2):
            print(f"[FAILED] fs_ctf FAILED for chunk {i} (rc={rc2}); see {args.log}")
            failed_chunks.append(i)

    print(f"Total chunks: {x}, Failed: {len(failed_chunks)}")


if __name__ == '__main__':
    main()
