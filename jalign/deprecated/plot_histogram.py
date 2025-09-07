#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : plot_histogram.py.py
# Time       ：2025/7/11 18:57
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
This script generates a histogram from a specified numeric column in either an LST or STAR file.
Features:
- Automatic detection of file format (.lst or .star)
- Optional use of EMAN2's StarFile3 parser if available
- Customizable histogram bin number
- Option to apply absolute value transformation to the data
Usage Examples:
    python plot_histogram.py -i particles.star -c _rlnLogLikeliContribution -b 20 -o llc.png
    python plot_histogram.py --input particles.lst --column score --bins 15 --abs
"""
import argparse
import re
import sys

try:
    from EMAN2 import StarFile3

    EMAN2_AVAILABLE = True
except ImportError:
    EMAN2_AVAILABLE = False

import matplotlib.pyplot as plt


def parse_lst_file(filepath, column_name):
    """Parses an LST file and extracts values for the specified column."""
    values = []
    column_pattern = re.compile(rf"{column_name}=(\S+)")

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = column_pattern.search(line)
            if match:
                val_str = match.group(1)
                try:
                    val = float(val_str)
                    values.append(val)
                except ValueError:
                    print(f"Non-numeric value encountered in column '{column_name}': '{val_str}'. Skipping.",
                          file=sys.stderr)
            else:
                print(f"Column '{column_name}' not found in line: {line}", file=sys.stderr)
    return values


def parse_star_manual(filepath, column_name):
    """Basic STAR file parser to extract values from a specified column."""
    values = []
    with open(filepath, 'r') as f:
        in_loop = False
        header = []
        col_idx = None
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower().startswith('loop_'):
                in_loop = True
                header = []
                col_idx = None
                continue
            if in_loop and line.startswith('_'):
                header.append(line.split()[0])
                if line.split()[0] == column_name:
                    col_idx = len(header) - 1
                continue
            if in_loop:
                parts = line.split()
                if col_idx is not None and col_idx < len(parts):
                    try:
                        values.append(float(parts[col_idx]))
                    except ValueError:
                        print(f"Non-numeric value encountered in column {column_name}: '{parts[col_idx]}'. Skipping.",
                              file=sys.stderr)
        return values


def parse_star_eman2(filepath, column_name):
    """Parse STAR file using EMAN2's StarFile3 parser."""
    data = StarFile3.read(filepath)
    for block in data:
        if column_name in block.keys():
            values = []
            for x in block[column_name]:
                try:
                    values.append(float(x))
                except ValueError:
                    print(f"Non-numeric value encountered in column {column_name}: '{x}'. Skipping.", file=sys.stderr)
            return values
    raise KeyError(f"Column {column_name} not found in STAR file via EMAN2")


def detect_file_type(filepath):
    if filepath.endswith('.lst'):
        return 'lst'
    elif filepath.endswith('.star'):
        return 'star'
    else:
        # Try to detect from contents
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip().startswith('#LST'):
                    return 'lst'
                if line.strip().startswith('data_') or '_rln' in line:
                    return 'star'
        raise ValueError("Cannot determine file type (LST or STAR).")


def main():
    parser = argparse.ArgumentParser(description="Plot histogram of a specified column from an LST or STAR file.")
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the input LST or STAR file.')
    parser.add_argument('-c', '--column', required=True,
                        help='Which column to plot (e.g., score, _rlnLogLikeliContribution).')
    parser.add_argument('-b', '--bins', type=int, default=10,
                        help='Number of histogram bins (default: 10).')
    parser.add_argument('--abs', action='store_true',
                        help='Use absolute value of the specified column (default: False).')
    parser.add_argument('-o', '--output', default=None,
                        help='Output image file (e.g., histogram.png). If not set, shows interactively.')
    args = parser.parse_args()

    column = args.column
    filetype = detect_file_type(args.input)

    if filetype == 'lst':
        values = parse_lst_file(args.input, column)
    else:
        if EMAN2_AVAILABLE:
            try:
                values = parse_star_eman2(args.input, column)
            except Exception as e:
                print(f"EMAN2 parsing failed ({e}), falling back to manual parser.", file=sys.stderr)
                values = parse_star_manual(args.input, column)
        else:
            values = parse_star_manual(args.input, column)

    if not values:
        print(f"No values found for column {column} in {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.abs:
        values = [abs(v) for v in values]

    # Plot histogram with bin edges as x-tick labels
    plt.figure(figsize=(10, 5))
    counts, bins, patches = plt.hist(values, bins=args.bins, edgecolor='black')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.xticks(bin_centers, [f"{b:.2e}" for b in bins[:-1]], rotation=45, ha='right')
    plt.tick_params(axis='x', length=0)  # 隐藏 x 轴的短线刻度

    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Number of particles')
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output)
        print(f"Histogram saved to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
