#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : delete_ogs.py
# Time       ：2025/9/27 3:20
# Author     ：Jago
# Email      ：huwl@hku.hk
# Description：
Delete specified optics-groups from matching_tomograms.star and matching.star, saving full backups (*_full.star) before modification.
For matching_tomograms.star
 - Remove rows in data_global and corresponding tomogram blocks matching given OGs.
 - Renumber remaining _rlnOpticsGroupName digit parts to 1..N in order of appearance.
For matching.star
 - Remove rows in data_optics and data_particles for given OGs.
 - Renumber remaining _rlnOpticsGroup and update _rlnOpticsGroupName digit parts to 1..N.
For both files, renumbering preserves order of first appearance.
A compact 3-line mapping table (old IDs, divider, new IDs) is printed.
Example:
    python delete_ogs.py -t matching_tomograms.star -m matching.star --og 3 5
"""
from pathlib import Path
import argparse
import sys
import re
from typing import Dict
import pandas as pd
import starfile
import shutil

width = shutil.get_terminal_size((80, 20)).columns


def extract_digits_int(s: str) -> int:
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else None


def replace_first_digit_group(name: str, new_num: int) -> str:
    s = str(name)
    m = re.search(r'(\d+)', s)
    if not m:
        return s
    return s[:m.start()] + str(new_num) + s[m.end():]


def print_mapping_table(mapping: Dict[int, int]):
    """Print mapping dict in three-line format: keys, pipes, values"""
    keys = list(mapping.keys())
    vals = list(mapping.values())
    keys_str = [str(k) for k in keys]
    vals_str = [str(v) for v in vals]
    # 每列宽度取 key/value 最大长度
    col_widths = [max(len(k), len(v)) for k, v in zip(keys_str, vals_str)]

    start = 0
    while start < len(keys):
        # 计算当前行能放多少列
        curr_width = 0
        end = start
        while end < len(keys):
            next_width = col_widths[end] + 1  # 每列之间加一个空格
            if curr_width + next_width > width and end > start:
                break
            curr_width += next_width
            end += 1

        # 构建当前行
        line_keys = " ".join(k.rjust(col_widths[i]) for i, k in enumerate(keys_str[start:end]))
        line_pipe = " ".join("|".rjust(col_widths[i]) for i in range(start, end))
        line_vals = " ".join(v.rjust(col_widths[i]) for i, v in enumerate(vals_str[start:end]))

        print(line_keys)
        print(line_pipe)
        print(line_vals)

        start = end


def renumber_global_names(df_global: pd.DataFrame):
    """Renumber rlnOpticsGroupName digits to 1..N in order of appearance"""
    seen = []
    old_to_new = {}
    for nm in df_global["rlnOpticsGroupName"].astype(str):
        old = extract_digits_int(nm)
        if old is not None and old not in seen:
            seen.append(old)
    for i, old in enumerate(seen, start=1):
        old_to_new[old] = i

    df_new = df_global.copy()
    df_new["rlnOpticsGroupName"] = df_new["rlnOpticsGroupName"].astype(str).apply(
        lambda nm: replace_first_digit_group(nm, old_to_new.get(extract_digits_int(nm))))
    return df_new, old_to_new


def delete_ogs_from_star(star_path: Path, delete_ogs: set):
    print(f"Processing {star_path.name}, deleting OGs: {sorted(delete_ogs)}")
    star = starfile.read(star_path, always_dict=True)

    # 保存 full 备份
    full_path = Path(f"{star_path.stem}_full.star")
    starfile.write(star, full_path)
    print(f"[INFO] Saved full backup: {full_path.name}")

    if "global" in star:
        # tomograms
        df_global = star["global"].copy()
        keep_mask = []
        to_remove_tomos = []
        for idx, row in df_global.iterrows():
            num = extract_digits_int(row["rlnOpticsGroupName"])
            if num in delete_ogs:
                keep_mask.append(False)
                to_remove_tomos.append(str(row["rlnTomoName"]))
            else:
                keep_mask.append(True)
        df_global_clean = df_global[keep_mask].reset_index(drop=True)
        star_clean = {k: v.copy() for k, v in star.items()}
        star_clean["global"] = df_global_clean

        # 删除对应 per-tomogram 块
        for tname in to_remove_tomos:
            if tname in star_clean:
                del star_clean[tname]

        # 重编号
        df_global_ren, mapping = renumber_global_names(df_global_clean)
        star_clean["global"] = df_global_ren
    else:
        # particles
        df_optics = star["optics"].copy()
        df_particles = star["particles"].copy()

        # 删除 optics 行
        keep_optics_mask = []
        for _, row in df_optics.iterrows():
            num = int(row["rlnOpticsGroup"])
            if num in delete_ogs:
                keep_optics_mask.append(False)
            else:
                keep_optics_mask.append(True)
        df_optics_clean = df_optics[keep_optics_mask].reset_index(drop=True)

        # 根据 rlnOpticsGroup 删除 particles 行
        kept_groups = set(df_optics_clean["rlnOpticsGroup"].astype(int).tolist())
        df_particles_clean = df_particles[df_particles["rlnOpticsGroup"].astype(int).isin(kept_groups)].reset_index(drop=True)

        # 重编号 optics & particles
        df_optics_new, mapping = renumber_global_names(df_optics_clean)
        df_optics_new["rlnOpticsGroup"] = df_optics_new["rlnOpticsGroup"].astype(int).map(lambda v: mapping.get(v, v))
        df_particles_new = df_particles_clean.copy()
        df_particles_new["rlnOpticsGroup"] = df_particles_new["rlnOpticsGroup"].astype(int).map(lambda v: mapping.get(v, v))

        star_clean = {"general": star["general"].copy(),
                      "optics": df_optics_new,
                      "particles": df_particles_new}

    # 写到当前路径下
    out_path = Path(star_path.name)
    starfile.write(star_clean, out_path)
    print(f"Wrote cleaned file: {out_path.name}")
    print_mapping_table(mapping)


def main():
    ap = argparse.ArgumentParser(
        description="Delete specified optics-groups from matching_tomograms.star and matching.star")
    ap.add_argument("-t", "--matching_tomograms", required=True, help="input matching_tomograms.star")
    ap.add_argument("-m", "--matching", required=True, help="input matching.star")
    ap.add_argument("--og", type=int, nargs="+", help="delete specified optics-group(s)")
    args = ap.parse_args()

    if not Path(args.matching_tomograms).exists():
        print(f"matching_tomograms.star not found: {args.matching_tomograms}", file=sys.stderr)
        sys.exit(2)
    if not Path(args.matching).exists():
        print(f"matching.star not found: {args.matching}", file=sys.stderr)
        sys.exit(2)

    delete_ogs_set = set(args.og) if args.og else set()

    print("------START------")
    if delete_ogs_set:
        delete_ogs_from_star(Path(args.matching_tomograms), delete_ogs_set)
        delete_ogs_from_star(Path(args.matching), delete_ogs_set)
    print("----- DONE -----")


if __name__ == "__main__":
    main()
