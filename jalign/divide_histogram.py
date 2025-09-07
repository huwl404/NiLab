#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : divide_histogram.py
# Time       ：2025/8/9 20:47
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
该脚本根据 LST 或 STAR 文件中指定的数值列绘制直方图，并将数据按区间拆分为多个新文件，或按数据升序等分为多个新文件。
功能：
自动识别输入文件格式（.lst 或 .star）
自定义直方图的分箱数量（bins）
根据直方图分箱结果自动生成对应的分箱数据文件
根据分箱数量生成含相同数量颗粒的新文件
使用示例：
python divide_histogram.py -i data/run_data.star -c _rlnLogLikeliContribution -o test -p split
python divide_histogram.py -i data/run_data.star -c _rlnLogLikeliContribution -b 5 --min 594874 --max 604751 --samesize -o test
python divide_histogram.py -i data/run64_data_s2_99_jalign.lst -c score -b 5 --samesize --abs -o test -p abs
"""
import argparse
import os
import re
import sys
from math import isfinite
import matplotlib.pyplot as plt


def detect_file_type(filepath):
    if filepath.endswith('.lst'):
        return 'lst'
    if filepath.endswith('.star'):
        return 'star'
    # try detect from content
    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('#LST'):
                return 'lst'
            if s.lower().startswith('data_') or s.startswith('loop_') or s.startswith('_'):
                return 'star'
    raise ValueError("Cannot determine file type (LST or STAR). Use .lst/.star extension or check file content.")


def parse_lst_lines(lines, column_name):
    """
    解析 LST 文件，返回header_lines [line, ...] 和 包含该列的行的元组列表 [(original_line, value), ...]。
    缺少该列或非数值的行将被跳过（并警告）。
    """
    pattern = re.compile(rf"{re.escape(column_name)}=(\S+)")
    entries = []
    header_lines = []
    for li, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            header_lines.append(raw.rstrip('\n'))
            continue
        m = pattern.search(raw)
        if not m:
            print(f"[LST] line {li+1}: column '{column_name}' not found -> skipping", file=sys.stderr)
            continue
        val_str = m.group(1)
        try:
            val = float(val_str)
            entries.append((raw.rstrip('\n'), val))
        except ValueError:
            print(f"[LST] line {li+1}: non-numeric value '{val_str}' for column '{column_name}' -> skipping", file=sys.stderr)
    return header_lines, entries


def find_loops_in_star(lines):
    """
    找出 STAR 文件中所有 loop_ 的位置与 header。
    返回 list of dict:
      {
        'loop_start': idx_of_loop_line,
        'header_start': idx_first_header_line (first line starting with '_'),
        'header_end': idx_last_header_line (exclusive, i.e. header lines are [header_start:header_end)),
        'data_start': idx_first_data_line,
        'data_end': idx_line_after_last_data (exclusive)
      }
    """
    n = len(lines)
    i = 0
    loops = []
    while i < n:
        line = lines[i].strip()
        if line.lower().startswith('loop_'):
            loop_start = i
            # gather headers
            j = i + 1
            header_start = None
            while j < n and lines[j].strip().startswith('_'):
                if header_start is None:
                    header_start = j
                j += 1
            header_end = j  # exclusive

            data_start = j
            k = j
            while k < n:
                s = lines[k].strip()
                if s.startswith('_') or s.lower().startswith('loop_') or s.lower().startswith('data_'):
                    break
                k += 1
            data_end = k  # exclusive
            loops.append({
                'loop_start': loop_start,
                'header_start': header_start,
                'header_end': header_end,
                'data_start': data_start,
                'data_end': data_end,
            })
            i = data_end
        else:
            i += 1
    return loops


def parse_star_loop_values(lines, loop):
    """
    给定 loop dict，返回 header lines list 和 data list
    """
    # headers are like '_rlnAnglePsi #1' include column number, but we used split()[0]
    header_lines = [ln.strip().split()[0] for ln in lines[loop['header_start']:loop['header_end']]]
    data = [ln.rstrip('\n') for ln in lines[loop['data_start']:loop['data_end']] if ln.strip() != '']
    return header_lines, data


def parse_star_lines(lines, column_name):
    """
    解析 STAR 文件，返回：
      - loops
      - target loop index in loops list
      - list of tuples (original_data_line, value)
    如果找不到 column，则抛出 KeyError。
    """
    loops = find_loops_in_star(lines)
    if not loops:
        raise ValueError("No loop_ blocks found in STAR file.")

    if not column_name.startswith('_'):
        column_name = '_' + column_name

    # 搜索包含列的 loop
    for li, loop in enumerate(loops):
        header_lines, data = parse_star_loop_values(lines, loop)
        if column_name in header_lines:
            col_idx = header_lines.index(column_name)
            entries = []
            for idx, dline in enumerate(data):
                parts = dline.split()
                if col_idx >= len(parts):
                    print(f"[STAR] loop {li}: data line {idx+1} has fewer columns than header -> skipping", file=sys.stderr)
                    continue
                val_str = parts[col_idx]
                try:
                    val = float(val_str)
                    entries.append((dline, val))
                except ValueError:
                    print(f"[STAR] loop {li} line {idx+1}: non-numeric value '{val_str}' -> skipping", file=sys.stderr)
            return loops, li, entries
    raise KeyError(f"Column '{column_name}' not found in any loop of STAR file.")


def assign_bins(entries, bins, vmin, vmax):
    """
    entries: list of tuples (orig_line, value)
    returns: list of lists, each inner list are entries in that bin in same tuple format, and list of edge points
    Bins are [b0,b1), [b1,b2), ..., [b_{n-1}, b_n]  (last bin inclusive right)
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")

    # compute edges
    step = (vmax - vmin) / bins
    edges = [vmin + i * step for i in range(bins + 1)]
    # place entries
    buckets = [[] for _ in range(bins)]
    skipped = 0

    for ent in entries:
        val = ent[1]
        if not isfinite(val):
            skipped += 1
            continue
        if val < vmin or val > vmax:
            skipped += 1
            continue

        # find bin index
        if val == vmax:
            bin_idx = bins - 1
        else:
            bin_idx = int((val - vmin) / step)
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx >= bins:
                bin_idx = bins - 1
        buckets[bin_idx].append(ent)

    print(f"Skipped particles (out of range or non-finite): {skipped}")
    return buckets, edges


def assign_bins_same_size(entries, bins, vmin, vmax):
    """按颗粒数均分成 bins 份，返回分组结果和边界值"""
    if bins < 1:
        raise ValueError("bins must be >= 1")

    # 先过滤
    in_range = [(entry, val) for entry, val in entries if vmin <= val <= vmax]
    skipped = len(entries) - len(in_range)
    print(f"Skipped particles (out of range or non-finite): {skipped}")

    if not in_range:
        raise ValueError("No particles within the specified range.")

    entries_sorted = sorted(in_range, key=lambda x: x[1])
    total = len(entries_sorted)

    idx_edges = [int(round(i * total / bins)) for i in range(bins + 1)]
    edges = [entries_sorted[i][1] if i < total else entries_sorted[-1][1] for i in idx_edges]

    buckets = []
    for i in range(bins):
        start, end = idx_edges[i], idx_edges[i + 1]
        buckets.append(entries_sorted[start:end])
    return buckets, edges


def save_histogram_plot(entries, edges, outpath, column_name, vertical_lines=None):
    """绘制直方图并保存"""
    values = [v for _, v in entries]
    plt.figure(figsize=(10, 5))
    counts, bins, patches = plt.hist(values, bins=edges, edgecolor='black')

    # 在每个柱子上方标注颗粒数
    for i in range(len(counts)):
        plt.text((bins[i] + bins[i+1]) / 2, counts[i], str(int(counts[i])), ha='center', va='bottom', fontsize=9)

    if vertical_lines:
        for v in vertical_lines[1:-1]:
            plt.axvline(v, color='red', linestyle='--', linewidth=1)
        for v in vertical_lines:
            plt.text(v, plt.ylim()[1] * 0.05, f'{v:.6f}', rotation=90,
                     ha='center', va='bottom', color='red', fontsize=9, backgroundcolor='white')
    else:
        for v in bins:
            plt.text(v, plt.ylim()[1] * 0.05, f'{v:.6f}', rotation=90,
                     ha='center', va='bottom', color='red', fontsize=9, backgroundcolor='white')

    plt.xlabel(column_name)
    plt.ylabel('Particle Count')
    plt.title(f'Histogram of {column_name}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved histogram to {outpath}.")


def write_lst_bucket_file(header_lines, bucket_entries, outpath):
    with open(outpath, 'w') as f:
        # write header
        for h in header_lines:
            f.write(h.rstrip('\n') + '\n')
        for line, _val in bucket_entries:
            f.write(line.rstrip('\n') + '\n')
    print(f"Wrote {len(bucket_entries)} lines to {outpath}")


def write_star_bucket_file(lines, loops, target_loop_idx, bucket_entries, outpath):
    loop = loops[target_loop_idx]
    data_lines = [ent[0] for ent in bucket_entries]

    with open(outpath, 'w') as f:
        for idx in range(0, loop['loop_start'] + 1):
            f.write(lines[idx])

        for idx in range(loop['header_start'], loop['header_end']):
            f.write(lines[idx])

        for ln in data_lines:
            f.write(ln.rstrip('\n') + '\n')

        f.write('\n')
        for idx in range(loop['data_end'], len(lines)):
            f.write(lines[idx])
    print(f"Wrote {len(bucket_entries)} particles to {outpath}")


def process_entries(file_type, entries, args, prefix, header=None, lines=None, loops=None, loop_index=None):
    if not entries:
        raise KeyError(f"No numeric entries found for column '{args.column}' in the '{args.input}'.")

    vals = [v for _, v in entries]
    computed_min = min(vals)
    computed_max = max(vals)
    vmin = args.vmin if args.vmin is not None else computed_min
    vmax = args.vmax if args.vmax is not None else computed_max
    if vmin >= vmax:
        print(f"Invalid range: min ({vmin}) must be < max ({vmax}).", file=sys.stderr)
        sys.exit(2)

    print(f"Input: {args.input}\n"
          f"Column: {args.column}\n"
          f"Bins: {args.bins}\n"
          f"Range: [{vmin}, {vmax}]")

    # 分桶逻辑
    if args.samesize:
        buckets, edges = assign_bins_same_size(entries, args.bins, vmin, vmax)
        vertical_lines = edges
    else:
        buckets, edges = assign_bins(entries, args.bins, vmin, vmax)
        vertical_lines = None

    total_included = sum(len(b) for b in buckets)
    print(f"Included particles: {total_included}")

    for i, b in enumerate(buckets):
        low, high = edges[i], edges[i + 1]
        right_bracket = "]" if i == len(buckets) - 1 else ")"
        print(f"Bin {i + 1}: [{low:.6g}, {high:.6g}{right_bracket} -> {len(b)} particles")

    if not args.onlyhist:
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        # 写文件
        for i, b in enumerate(buckets):
            outpath = os.path.join(args.output_dir, f"{prefix}_bin{i + 1}.{file_type}")
            if file_type == "lst":
                write_lst_bucket_file(header, b, outpath)
            else:
                write_star_bucket_file(lines, loops, loop_index, b, outpath)

    hist_path = os.path.join(args.output_dir, f"{prefix}_histogram.png")
    # 绘制直方图
    save_histogram_plot(entries, edges, hist_path, args.column, vertical_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Divide LST/STAR by column name parameter and save per-bucket files.")
    parser.add_argument('-i', '--input', required=True,
                        help='Input LST or STAR file path.')
    parser.add_argument('-c', '--column', required=True,
                        help='Column name to use (e.g. _rlnLogLikeliContribution or score).')
    parser.add_argument('-b', '--bins', type=int, default=10,
                        help='Number of histogram bins (default: 10).')
    parser.add_argument('--min', dest='vmin', type=float, default=None,
                        help='Minimum value to consider (inclusive).')
    parser.add_argument('--max', dest='vmax', type=float, default=None,
                        help='Maximum value to consider (inclusive).')
    parser.add_argument('--samesize', action='store_true',
                        help='Divide particles with same size by ascending order of the column name (default: False).')
    parser.add_argument('--onlyhist', action='store_true',
                        help='Only save histogram picture, don\'t save particles files (default: False).')
    parser.add_argument('--abs', dest='use_abs', action='store_true',
                        help='Use absolute value of the chosen column before binning (default: False).')
    parser.add_argument('-o', '--output-dir', default='.',
                        help='Directory to save output files (default: current folder).')
    parser.add_argument('-p', '--prefix', default=None,
                        help='Output file prefix (default: <input basename>_bin / <input basename>_histogram).')
    args = parser.parse_args()

    try:
        ftype = detect_file_type(args.input)
    except Exception as e:
        print(f"File format detect failed: {e}.", file=sys.stderr)
        sys.exit(2)

    if args.prefix:
        prefix = args.prefix
    else:
        prefix = os.path.splitext(os.path.basename(args.input))[0]

    with open(args.input, 'r') as f:
        lines = f.readlines()

    if ftype == 'lst':
        header_lines, entries = parse_lst_lines(lines, args.column)
        # 使用绝对值分桶，写文本文件时仍使用原始行
        if args.use_abs:
            entries = [(ln, abs(v)) for (ln, v) in entries]
            print(f"Converted column '{args.column}' to absolute values before binning.")
        process_entries("lst", entries, args, prefix, header=header_lines)
    else:
        loops, loop_index, entries = parse_star_lines(lines, args.column)
        if args.use_abs:
            entries = [(ln, abs(v)) for (ln, v) in entries]
            print(f"Converted column '{args.column}' to absolute values before binning.")
        process_entries("star", entries, args, prefix, lines=lines, loops=loops, loop_index=loop_index)

    print("Done.")


if __name__ == '__main__':
    main()
