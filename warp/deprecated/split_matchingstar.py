#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : split_matchingstar.py
# Time       ：2025/9/26 23:08
# Author     ：Jago
# Email      ：huwl@hku.hk #
Description：
Split matching_tomograms.star and matching.star into two optimisation sets by optics-group range, fixing per-tilt dose
if requested and renumbering optics groups to 1-based in outputs.
  1. For each input matching_tomograms.star the script:
  - optionally fix `_rlnTomoImportFractionalDose` in the `data_global` block to the user-specified dose (written with 3 decimals).
    If any rows are changed the original full tomograms star is saved as `<basename>_original.star` and the modified full file overwrites the original filename.
  2. Split the `data_global` rows (preserving original order) into two halves (first ceil(N/2), rest).
  - for each half it keeps only the per-tomogram data blocks that correspond to the tomogram names in that half,
  - computes the optics-group number range from `_rlnOpticsGroupName`,
  - renumbers those optics-group names so their digit parts start from 1,
  - writes two tomograms files named: <matching_tomograms>_OG<start>-<end>.star,
  - prints the mapping old->new (three-line table) for each produced tomograms file.
  3. Read the input matching.star and split it into two particle files that correspond to the optics groups in each tomograms half:
  - Keep `data_general` in both outputs
  - `data_optics` & `data_particles` block : keep only rows whose original `_rlnOpticsGroup` belong to that half,
  - Renumber optics-group IDs and the digit part of `_rlnOpticsGroupName` to start at 1 in each split,
  - Write two particle star files named: <matching>_OG<start>-<end>.star,
  - prints the mapping old->new (three-line table) for each produced particle file.
  4. For each pair of outputs (tomograms + particles) write a  optimisation-set file:
  - matching_optimisation_set_OG<start>-<end>.star which contains two entries: _rlnTomoParticlesFile &_rlnTomoTomogramsFile.
Examples:
  Fix dose in matching_tomograms.star, split and generate optimisation sets:
      python split_matchingstar.py -t matching_tomograms.star -m matching.star -d 3.0
  Split without changing dose:
      python split_matchingstar.py -t matching_tomograms.star -m matching.star
"""
from pathlib import Path
import argparse
import sys
import re
import math
from typing import Dict, Tuple
import starfile
import pandas as pd
import shutil

width = shutil.get_terminal_size((80, 20)).columns


def extract_digits_int(s: str) -> int:
    """Extract first continuous digit group from string, return int or None."""
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else None


def replace_first_digit_group(name: str, new_num: int) -> str:
    """Replace first continuous digit group in name by new_num, preserve other text."""
    s = str(name)
    m = re.search(r'(\d+)', s)
    if not m:
        return s
    return s[:m.start()] + str(new_num) + s[m.end():]


def get_og_range(df_g: pd.DataFrame) -> Tuple[int, int]:
    names = df_g["rlnOpticsGroupName"].dropna().astype(str).tolist()
    nums = [extract_digits_int(s) for s in names]
    nums = [n for n in nums if n is not None]
    if not nums:
        return 0, 0
    return min(nums), max(nums)


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


def renumber_global_names(df_global: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Renumber rlnOpticsGroupName entries in data_global so that extracted digits map to 1..N in order
    of appearance. Returns (new_df_global, mapping_oldnum->newnum).
    """
    names = df_global["rlnOpticsGroupName"].astype(str).tolist()
    seen = []
    old_to_new = {}
    for nm in names:
        old = extract_digits_int(nm)
        if old is None:
            continue
        if old not in seen:
            seen.append(old)

    for i, old in enumerate(seen, start=1):
        old_to_new[old] = i

    df_new = df_global.copy()

    def _map_name(nm):
        old = extract_digits_int(nm)
        if old is None:
            return str(nm)
        new = old_to_new.get(old, None)
        if new is None:
            return str(nm)
        return replace_first_digit_group(nm, new)

    df_new["rlnOpticsGroupName"] = df_new["rlnOpticsGroupName"].astype(str).apply(_map_name)
    return df_new, old_to_new


def renumber_optics_and_particles(df_optics: pd.DataFrame, df_particles: pd.DataFrame) \
        -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    """
    Renumber optics groups in df_optics and df_particles so rlnOpticsGroup becomes 1..N.
    Also replace first digit group in rlnOpticsGroupName accordingly.
    Returns (df_optics_new, df_particles_new, mapping_oldnum->newnum).
    """
    names = df_optics["rlnOpticsGroupName"].astype(str).tolist()
    seen_old_nums = []
    for nm in names:
        old = extract_digits_int(nm)
        if old is None:
            continue
        if old not in seen_old_nums:
            seen_old_nums.append(old)

    mapping = {old: i for i, old in enumerate(seen_old_nums, start=1)}

    df_opt_new = df_optics.copy()
    df_opt_new["rlnOpticsGroup"] = df_opt_new["rlnOpticsGroup"].astype(int).map(lambda v: mapping.get(int(v), v))
    # 如果 mapping 里有这个旧编号 → 返回新的编号；如果没有 → 就退回原编号；如果提取失败 → 用 0。
    # lambda nm: replace_first_digit_group(nm, mapping.get(extract_digits_int(nm), extract_digits_int(nm) or 0))
    df_opt_new["rlnOpticsGroupName"] = df_opt_new["rlnOpticsGroupName"].astype(str).apply(
        lambda nm: replace_first_digit_group(nm, mapping.get(extract_digits_int(nm))))

    df_part_new = df_particles.copy()
    df_part_new["rlnOpticsGroup"] = df_part_new["rlnOpticsGroup"].astype(int).map(lambda v: mapping.get(int(v), v))

    return df_opt_new, df_part_new, mapping


def fix_dose_split_tomograms(in_path: Path, dose) -> Tuple[Path, Path, Dict[int, int], Dict[int, int]]:
    """
    Read matching_tomograms.star, fix fractional dose, split into two tomograms star files.
    Returns (out_path1, out_path2, (og1_start, og1_end), (og2_start, og2_end))
    """
    print(f"Reading tomograms star: {in_path}")
    star = starfile.read(in_path, always_dict=True)
    df_global = star["global"].copy()

    if dose is not None:
        # check / fix fractional dose
        changed_rows = []
        # compare with tolerance because floats may have formatting
        for idx, val in df_global["rlnTomoImportFractionalDose"].items():
            try:
                cur = float(val)
            except Exception:
                cur = None

            if cur is None or round(cur, 3) != round(dose, 3):
                df_global.at[idx, "rlnTomoImportFractionalDose"] = round(dose, 3)
                changed_rows.append((idx, str(df_global.at[idx, "rlnTomoName"]), cur))

        if changed_rows:
            print(f"[INFO] Updated {len(changed_rows)} rlnTomoImportFractionalDose entries to {dose:.3f}:")
            for idx, name, old in changed_rows:
                print(f"  row {idx}: tomogram '{name}' old={old} -> new={dose:.3f}")

            # first save original (full) star as *_original.star
            orig_copy = Path(f"{in_path.stem}_original.star")
            starfile.write(star, orig_copy)
            print(f"[INFO] Original tomograms file saved as: {orig_copy.name}")

            # write the modified full matching_tomograms.star (with updated global) using original filename
            star_mod = star.copy()
            star_mod["global"] = df_global
            starfile.write(star_mod, in_path)
            print(f"[INFO] Modified full tomograms star (dose-fixed) written as: {in_path}")
        else:
            print("[INFO] All fractional dose entries already match target value.")

    # split data_global rows into two halves preserving order
    tomonames = list(df_global["rlnTomoName"].astype(str).tolist())
    half = math.ceil(len(tomonames) / 2)
    half1_names, half2_names = tomonames[:half], tomonames[half:]

    all_block_keys = list(star.keys())
    dict1, dict2 = {}, {}
    dict1["global"] = df_global[df_global["rlnTomoName"].astype(str).isin(half1_names)].copy()
    dict2["global"] = df_global[df_global["rlnTomoName"].astype(str).isin(half2_names)].copy()
    # iterate original keys, keep blocks that correspond to any name
    for k in all_block_keys:
        if k == "global":
            continue
        if k in half1_names:
            dict1[k] = star[k].copy()
        elif k in half2_names:
            dict2[k] = star[k].copy()
        else:
            print(f"[SKIP] Tomograms star file containing rlnTomoName {k} not in global block.")

    s1, e1 = get_og_range(dict1["global"])
    s2, e2 = get_og_range(dict2["global"])

    base = in_path.stem
    # out1_path = Path(f"{base}_OG{og1_start}-{og1_end}.star")
    # out2_path = Path(f"{base}_OG{og2_start}-{og2_end}.star")
    # if input filename already contains _OG<start>-<end>, offset the new OG numbers
    m = re.search(r"_OG(\d+)-(\d+)$", base)
    if m:
        orig_start = int(m.group(1))
        prefix = base[:m.start()]
        s1, e1 = orig_start + s1 - 1, orig_start + e1 - 1
        s2, e2 = orig_start + s2 - 1, orig_start + e2 - 1
    else:
        prefix = base

    out1_path = Path(f"{prefix}_OG{s1}-{e1}.star")
    out2_path = Path(f"{prefix}_OG{s2}-{e2}.star")

    # renumber optics names in each global block so digit parts start at 1
    dict1_global_new, map1 = renumber_global_names(dict1["global"])
    dict2_global_new, map2 = renumber_global_names(dict2["global"])
    dict1["global"] = dict1_global_new
    dict2["global"] = dict2_global_new

    starfile.write(dict1, out1_path)
    starfile.write(dict2, out2_path)

    print(f"\nTomograms split -> {out1_path.name}")
    print_mapping_table(map1)
    print(f"\nTomograms split -> {out2_path.name}")
    print_mapping_table(map2)
    return out1_path, out2_path, map1, map2


def write_optim_set(tomograms_file: Path, particles_file: Path, s: int, e: int) -> Path:
    fname = Path(f"matching_optimisation_set_OG{s}-{e}.star")
    with fname.open("w", newline="") as fh:
        fh.write("\ndata_\n\n")
        fh.write(f"_rlnTomoParticlesFile   {particles_file.name}\n")
        fh.write(f"_rlnTomoTomogramsFile   {tomograms_file.name}\n")
    return fname


def split_particles_generate_sets(in_path: Path, tomograms_out1: Path, tomograms_out2: Path,
                                  map1: Dict[int, int], map2: Dict[int, int]):
    """
    Read matching.star, split into two files corresponding to optics groups extracted from tomograms outputs.
    Generate corresponding optimisation set files.
    """
    print(f"\nReading particles star: {in_path}")
    star = starfile.read(in_path, always_dict=True)
    df_general = star["general"].copy()
    df_optics = star["optics"].copy()
    df_particles = star["particles"].copy()

    set1, set2 = [], []
    for _, row in df_optics.iterrows():
        old_id = int(row["rlnOpticsGroup"])
        if old_id in map1:
            set1.append(old_id)
        elif old_id in map2:
            set2.append(old_id)
        else:
            print(f"[SKIP] Particles file containing non-existing original rlnOpticsGroup {old_id} in original tomograms file.")

    # prepare data_optics filtered frames
    df_optics1 = df_optics[df_optics["rlnOpticsGroup"].astype(int).isin(set1)].copy()
    df_optics2 = df_optics[df_optics["rlnOpticsGroup"].astype(int).isin(set2)].copy()

    df_particles1 = df_particles[df_particles["rlnOpticsGroup"].astype(int).isin(set1)].copy()
    df_particles2 = df_particles[df_particles["rlnOpticsGroup"].astype(int).isin(set2)].copy()

    s1, e1 = min(set1), max(set1)
    s2, e2 = min(set2), max(set2)
    base = in_path.stem
    # out1_path = Path(f"{base}_OG{s1}-{e1}.star")
    # out2_path = Path(f"{base}_OG{s2}-{e2}.star")
    m = re.search(r"_OG(\d+)-(\d+)$", base)
    if m:
        orig_start = int(m.group(1))
        prefix = base[:m.start()]
        s1, e1 = orig_start + s1 - 1, orig_start + e1 - 1
        s2, e2 = orig_start + s2 - 1, orig_start + e2 - 1
    else:
        prefix = base

    out1_path = Path(f"{prefix}_OG{s1}-{e1}.star")
    out2_path = Path(f"{prefix}_OG{s2}-{e2}.star")

    # dict1 = {"general": df_general.copy(), "optics": df_optics1, "particles": df_particles1}
    # dict2 = {"general": df_general.copy(), "optics": df_optics2, "particles": df_particles2}
    # renumber optics and particles groups to start from 1 for each split
    opt1_new, part1_new, mapping1 = renumber_optics_and_particles(df_optics1, df_particles1)
    opt2_new, part2_new, mapping2 = renumber_optics_and_particles(df_optics2, df_particles2)

    dict1 = {"general": df_general.copy(), "optics": opt1_new, "particles": part1_new}
    dict2 = {"general": df_general.copy(), "optics": opt2_new, "particles": part2_new}

    starfile.write(dict1, out1_path)
    starfile.write(dict2, out2_path)
    print(f"Particles split -> {out1_path.name}")
    print_mapping_table(mapping1)
    print(f"\nParticles split -> {out2_path.name}")
    print_mapping_table(mapping2)

    # write matching_optimisation_set files for both
    os1_path = write_optim_set(tomograms_out1, out1_path, s1, e1)
    os2_path = write_optim_set(tomograms_out2, out2_path, s2, e2)
    print(f"\nWrote optimisation set files -> {os1_path.name}, {os2_path.name}")


def main():
    ap = argparse.ArgumentParser(
        description="Split equally matching_tomograms.star and matching.star into two optimisation sets by "
                    "optics-group range and re-number optics-group as 1-based.")
    ap.add_argument("-t", "--matching_tomograms", required=True, help="input matching_tomograms.star")
    ap.add_argument("-m", "--matching", required=True, help="input matching.star")
    ap.add_argument("-d", "--dose", type=float, help="per-tilt dose.")
    args = ap.parse_args()

    tomograms_path = Path(args.matching_tomograms)
    matching_path = Path(args.matching)
    dose = float(args.dose) if args.dose is not None else None

    if not tomograms_path.exists():
        print(f"matching_tomograms not found: {tomograms_path}", file=sys.stderr)
        sys.exit(2)
    if not matching_path.exists():
        print(f"matching.star not found: {matching_path}", file=sys.stderr)
        sys.exit(2)

    print("------START------")
    tom_out1, tom_out2, tom_map1, tom_map2 = fix_dose_split_tomograms(tomograms_path, dose)
    split_particles_generate_sets(matching_path, tom_out1, tom_out2, tom_map1, tom_map2)
    print("----- Done -----")


if __name__ == "__main__":
    main()
