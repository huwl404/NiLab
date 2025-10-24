#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : batch_assembler.py
# Time       : 2025/10/24 11:30
# Author     : Jago
# Email      : huwl@hku.hk
# Description：
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET


def find_prefixes(parent: Path, base: str) -> List[str]:
    prefixes = set()
    for p in parent.iterdir():
        name = p.name
        if name == base:
            continue
        if name.endswith(base):
            prefix = name[: -len(base)]
            prefixes.add(prefix)
    return sorted(prefixes)


def ensure_all_exist(parent: Path, prefixes: List[str], folder_data: str, folder_processing: str, output: str) -> List[str]:
    missing = []
    for prefix in prefixes:
        items = [parent / (prefix + folder_data), parent / (prefix + folder_processing), parent / (prefix + output)]
        for it in items:
            if not it.exists():
                missing.append(str(it))
    return missing


def replace_param_in_xml_text(xml_text: str, param_name: str, new_value: str) -> str:
    """替换 <Param Name="param_name" Value="..." /> 中的 Value 值为 new_value（只替换第一个匹配项）"""
    # \s+ at least 1 space, \s* allow zero space
    pattern = re.compile(r'(<Param\s+Name="' + re.escape(param_name) + r'"\s+Value=")([^"]*)("\s*/>)')
    new_text, n = pattern.subn(r'\1' + new_value + r'\3', xml_text, count=1)
    return new_text


def replace_param(xml: Path, new_value_1: str, new_value_2: str, output: Path):
    tree = ET.parse(xml)
    root = tree.getroot()
    for param in root.findall(".//Param"):
        if param.get("Name") == "DataFolder":
            param.set("Value", new_value_1)
        elif param.get("Name") == "ProcessingFolder":
            param.set("Value", new_value_2)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def main():
    ap = argparse.ArgumentParser(description='Assemble warp outputs from prefixed chunks')
    ap.add_argument('--folder_data', required=True, help='folder with image files (e.g. frames)')
    ap.add_argument('--folder_processing', required=True, help='base folder for processing outputs (e.g. warp_frameseries)')
    ap.add_argument('--output', required=True, help='base output settings filename (e.g. warp_frameseries.settings)')
    args = ap.parse_args()

    fd_path = Path(args.folder_data)

    base_fd = fd_path.name
    parent = fd_path.parent
    base_fp = args.folder_processing
    base_out = args.output

    prefixes = find_prefixes(parent, base_fd)
    if not prefixes:
        print("[ERROR] No prefixed items found.")
        sys.exit(2)
    print(f"[INFO] Found prefixes: {prefixes} in {parent}")

    missing = ensure_all_exist(parent, prefixes, base_fd, base_fp, base_out)
    if missing:
        missing_str = "\n".join(missing)
        print(f"[ERROR] Following files or directories not found: {missing_str}")
        sys.exit(2)

    # copy first warp_frameseries.settings to warp_frameseries, then replace Param DataFolder and ProcessingFolder
    src_out = parent / (prefixes[0] + base_out)
    dst_out = parent / base_out
    # shutil.copy2(src_out, dst_out)
    # txt = dst_out.read_text(encoding='utf-8')
    # # replace DataFo      lder param to base_fd
    # txt = replace_param_in_xml_text(txt, 'DataFolder', base_fd)
    # # replace ProcessingFolder param to base_fp
    # txt = replace_param_in_xml_text(txt, 'ProcessingFolder', base_fp)
    # dst_out.write_text(txt, encoding='utf-8')
    replace_param(src_out, base_fd, base_fp, dst_out)

    target_proc = parent / base_fp
    target_proc.mkdir(parents=True, exist_ok=True)

    # copy first prefix's ctf_movies.settings to warp_frameseries, then replace Param DataFolder and ProcessingFolder
    src_ctf = parent / (prefixes[0] + base_fp) / "ctf_movies.settings"
    dst_ctf = target_proc / "ctf_movies.settings"
    replace_param(src_ctf, base_fd, base_fp, dst_ctf)

    target_log = target_proc / "logs"
    target_ps = target_proc / "powerspectrum"
    target_log.mkdir(parents=True, exist_ok=True)
    target_ps.mkdir(parents=True, exist_ok=True)
    # iterate over all prefixes' processing dirs: find xmls, replace prefix+base_fd -> base_fd, move xmls to target_proc
    for prefix in prefixes:
        src_proc = parent / (prefix + base_fp)
        xml_files = list(src_proc.rglob('*.xml'))
        for xf in xml_files:
            txt = xf.read_text(encoding='utf-8')
            new_txt = txt.replace(prefix + base_fd, base_fd)
            dst = target_proc / xf.name
            dst.write_text(new_txt, encoding='utf-8')
        print(f"[INFO] Copied and processed {len(xml_files)} xml files in {src_proc} to {target_proc}")

        log_folder = src_proc / "logs"
        for log in log_folder.iterdir():
            dest = target_log / log.name
            if dest.exists():
                os.remove(dest)
            shutil.move(log, dest)
        print(f"[INFO] Moved logs directories in {src_proc} to {target_log}")

        ps_folder = src_proc / "powerspectrum"
        for ps in ps_folder.iterdir():
            dest = target_ps / ps.name
            if dest.exists():
                os.remove(dest)
            shutil.move(ps, dest)
        print(f"[INFO] Moved powerspectrum directories in {src_proc} to {target_ps}")

    # merge processed_items.json files from every prefixed processing folder
    merged_data = []
    for prefix in prefixes:
        jf = parent / (prefix + base_fp) / 'processed_items.json'
        with jf.open("r", encoding="utf-8") as f:
            data = json.load(f)
            merged_data.extend(data)

    out_json_path = target_proc / 'processed_items.json'
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Merged processed_items.json files to {out_json_path}")

    # delete prefixed folder_data
    for p in prefixes:
        pfd = parent / (p + base_fd)
        shutil.rmtree(pfd)
        pfp = parent / (p + base_fp)
        shutil.rmtree(pfp)
        os.remove(parent / (p + base_out))
    print("[INFO] Deleted all batch files.")
    print("Done.")


if __name__ == '__main__':
    main()
