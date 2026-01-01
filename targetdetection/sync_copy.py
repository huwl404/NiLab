#!/usr/bin/env python
# -*- coding:utf-8 -*-

import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def sync_files(args):
    dir1 = Path(args.dir1)
    dir2 = Path(args.dir2)
    out_dir = Path(args.output)

    ext1 = args.ext1 if args.ext1.startswith('.') else f'.{args.ext1}'
    ext2 = args.ext2 if args.ext2.startswith('.') else f'.{args.ext2}'

    # 1. 检查输入路径
    if not dir1.is_dir() or not dir2.is_dir():
        print(f"[ERROR] One of the input directories does not exist.")
        return

    # 2. 获取参考文件夹中的所有文件名（不含后缀）
    # 使用 set 提高查找效率
    print(f"[INFO] Scanning reference folder: {dir1}")
    ref_names = {f.stem for f in dir1.glob(f"*{ext1}")}

    if not ref_names:
        print(f"[WARNING] No files with extension {ext1} found in {dir1}")
        return
    print(f"[INFO] Found {len(ref_names)} reference files.")

    # 3. 创建输出文件夹
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. 在文件夹 2 中寻找匹配的文件并拷贝
    print(f"[INFO] Scanning source folder: {dir2}")
    target_files = list(dir2.glob(f"*{ext2}"))

    copy_count = 0
    for f in tqdm(target_files, desc="Syncing", unit="file"):
        if f.stem in ref_names:
            # 执行拷贝 (copy2 会保留元数据)
            shutil.copy2(f, out_dir / f.name)
            copy_count += 1

    print(f"[INFO] Success! Copied {copy_count} files to {out_dir.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy files from Dir2 that match filenames in Dir1")
    parser.add_argument("-d1", "--dir1", type=str, required=True, help="Reference directory (folder 1)")
    parser.add_argument("-d2", "--dir2", type=str, required=True, help="Source directory to copy from (folder 2)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    parser.add_argument("-e1", "--ext1", type=str, default=".txt", help="Extension in folder 1 (default: .txt)")
    parser.add_argument("-e2", "--ext2", type=str, default=".png", help="Extension in folder 2 (default: .png)")

    args = parser.parse_args()
    sync_files(args)
