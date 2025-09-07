#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : lst2star.py
# Time       ：2025/8/20 21:14
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
该脚本根据 Jalign 的 .lst 输出文件筛选粒子 并写入一个新的 RELION .star 文件。
功能：
读取 Jalign 输出的 .lst 文件，其中每行包含类似 <param>=<value> 的字段。
自动定位并读取对应的 lst2 文件（Jalign 输入），将 .lst 中的行拼接为一个 tmp.lst。
根据用户指定的参数名与阈值条件（如 --column score --greaterthan 0.5），筛选符合条件的行。
将这些行转换为 RELION 风格的 _rlnImageName 标签（如 "000002@Extract/job050/10727/Images/00038_1_0.mrcs"）。
从输入的 .star 文件中过滤掉未入选的粒子，保留其它非粒子 block，并写出一个新的 .star 文件。
使用示例：
python lst2star.py --lst test/abs1.lst --star data/run64_data_s2.star
python lst2star.py --lst test/abs1.lst --star data/run64_data_s2.star --output test/s2_1stof5.star
python lst2star.py --lst data/run64_data_s2_99_jalign.lst --star data/run64_data_s2.star --output test/manual.star --column score --greaterthan 0.02 --lessthan 0.08 --abs
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
from EMAN2star import StarFile3


def merge_lst(lst1_path: str) -> str:
    """
    从 lst1 中解析出对应的 lst2 文件，拼接为 tmp.lst，返回 tmp.lst 的路径。
    支持 lst1 第二列包含多个不同的 lst2 文件名。
    """
    with open(lst1_path, 'r') as f1:
        lines1 = [l.strip() for l in f1 if l.strip() and not l.startswith('#')]

    if not lines1:
        sys.exit(f"{lst1_path}: empty or contains only comments.")

    # 缓存已读取的 lst2 文件，避免重复打开
    lst2_cache: dict[str, list[str]] = {}

    tmp_path = os.path.join(os.path.dirname(lst1_path), "tmp.lst")
    written = 0

    with open(tmp_path, 'w') as out:
        for line1 in lines1:
            parts = line1.split()
            if len(parts) < 2:
                print(f"{lst1_path} skipping line {line1}: malformed.", file=sys.stderr)
                continue

            idx_str, file_from_lst1 = parts[0], parts[1]

            try:
                idx = int(idx_str)
            except ValueError:
                print(f"{lst1_path} skipping line {line1}: bad index.", file=sys.stderr)
                continue

            # 加载 lst2 文件到缓存
            if file_from_lst1 not in lst2_cache:
                lst2_path = os.path.join(os.path.dirname(lst1_path), file_from_lst1)
                if not os.path.exists(lst2_path):
                    print(f"{lst1_path} skipping line {line1}: {lst2_path} not found.", file=sys.stderr)
                    continue

                with open(lst2_path, 'r') as f2:
                    lst2_lines = [l.strip() for l in f2 if l.strip() and not l.startswith('#')]
                lst2_cache[file_from_lst1] = lst2_lines

            lines2 = lst2_cache[file_from_lst1]

            if idx < 0 or idx >= len(lines2):
                print(f"{lst1_path} skipping line {line1}: index {idx} out of range in {file_from_lst1}.", file=sys.stderr)
                continue

            merged = lines2[idx] + "\t" + "\t".join(parts[2:])
            out.write(merged + "\n")
            written += 1

    print(f"Generated merged tmp.lst with {written} lines at {tmp_path}", file=sys.stderr)
    return tmp_path


def read_lst(path: str, param: str | None, greaterthan: float | None, lessthan: float | None, use_abs: bool = False) -> list:
    """
    从 tmp.lst 读取，返回符合条件的 rlnImageName 列表。
    """
    kept = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue

            parts = s.split()
            if len(parts) < 2:
                print(f"{path} skipping line {s}: invalid format, missing second column.", file=sys.stderr)
                continue

            try:
                image_id = int(parts[0])
            except ValueError:
                print(f"{path} skipping line {s}: first token not int.", file=sys.stderr)
                continue

            micro_path = parts[1]
            # EMAN2 image id is 0-based, RELION uses 1-based
            new_image_id = image_id + 1
            tag = f"{str(new_image_id).zfill(6)}@{micro_path}"

            if param is None and greaterthan is None and lessthan is None:
                kept.append(tag)
                continue

            # otherwise try to find param=value among tokens
            val = None
            needle = f"{param}="
            for token in parts[2:]:
                if token.startswith(needle):
                    raw = token.split('=', 1)[1]
                    try:
                        v = float(raw)
                        val = abs(v) if use_abs else v
                    except ValueError:
                        print(f"{path} skipping line {s}: cannot parse numeric value for {param}.", file=sys.stderr)
                        val = None
                    break

            if val is None:
                print(f"{path} skipping line {s}: '{param}' not found or non-numeric.", file=sys.stderr)
                continue

            ok_gt = True if greaterthan is None else (val > greaterthan)
            ok_lt = True if lessthan is None else (val < lessthan)
            if ok_gt and ok_lt:
                kept.append(tag)

    print(f"Kept {len(kept)} usable lines from {path}.", file=sys.stderr)
    return kept


def filter_star(in_star: str, out_star: str, keep_images: list) -> int:
    """
    过滤 STAR 文件，保留 keep_images 中的粒子。
    """
    star = StarFile3(in_star)
    output_blocks = {}
    kept_num = 0

    for block_name, block in star.items():
        if 'rlnImageName' in block:
            image_names = np.array(block['rlnImageName']).astype(str)
            mask = np.isin(image_names, keep_images)
            kept_indices = np.nonzero(mask)[0]
            kept_num += len(kept_indices)

            if len(kept_indices) == 0:
                print(f"{in_star} skipping block '{block_name}': no matching particles kept.", file=sys.stderr)
                continue

            # filter each column by kept_indices
            filtered_block = {}
            for k, v in block.items():
                arr = np.array(v) if not isinstance(v, np.ndarray) else v
                filtered_block[k] = arr[kept_indices].tolist()
            output_blocks[block_name] = filtered_block
        else:
            # non-particle block, copy as-is
            output_blocks[block_name] = block

    # write STAR text
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

    print(f"Read {kept_num} usable lines from {in_star}.", file=sys.stderr)
    return kept_num


def main():
    p = argparse.ArgumentParser(
        description="Filter Relion .star file using Jalign .lst results.")
    p.add_argument('--lst', required=True,
                   help="Jalign output .lst file (contains evaluation column).")
    p.add_argument('--star', required=True,
                   help="Input RELION .star file to be filtered.")
    p.add_argument('--output', required=False, default=None,
                   help="Output .star file (default: <lst basename>.star in the same directory with lst file).")
    p.add_argument('--column', required=False, default=None,
                   help="Column name in lst1 to filter on (e.g. score). If omitted, all lst particles are kept.")
    p.add_argument('--greaterthan', type=float, default=None,
                   help="Keep particles with param > GREATERTHAN (optional).")
    p.add_argument('--lessthan', type=float, default=None,
                   help="Keep particles with param < LESSTHAN (optional).")
    p.add_argument('--abs', dest='use_abs', action='store_true',
                   help="Use absolute value of the chosen column before binning (default: False).")
    args = p.parse_args()

    if args.column is None and (args.greaterthan is not None or args.lessthan is not None):
        sys.exit(f"Must assign column name if you assign filter threshold.")

    if args.output:
        output = args.output
    else:
        filename = os.path.splitext(os.path.basename(args.lst))[0] + ".star"
        output = os.path.join(os.path.dirname(args.lst), filename)

    tmp_lst = merge_lst(args.lst)

    kept_images = read_lst(tmp_lst, args.column, args.greaterthan, args.lessthan, args.use_abs)

    if not kept_images:
        sys.exit(f"No images kept in {args.lst}, check your Jalign output/input lst file.")

    kept_num = filter_star(args.star, output, kept_images)
    if not kept_num:
        sys.exit(f"No particles kept in {args.star}, check your star file.")
    print(f"Wrote {kept_num} particles to {output}.")


if __name__ == '__main__':
    main()
