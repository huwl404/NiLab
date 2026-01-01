#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from pathlib import Path
from tqdm import tqdm


def run_filter(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    ids_file = Path(args.ids)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Input folder not found: {input_dir}")
        return

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"[ERROR] No txt files found in {input_dir}")
        return
    print(f"[INFO] Processing {len(txt_files)} files...")

    if not ids_file.exists():
        print(f"[ERROR] ID file not found: {ids_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(ids_file, 'r') as f:
        target_ids = {int(line.strip()) for line in f if line.strip()}
    print(f"[INFO] Loaded {len(target_ids)} target global IDs.")

    current_global_idx = args.lbl_idx_offset + 1
    out_count = 0

    for txt_path in tqdm(txt_files, desc="Processing"):
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 判断当前行是否在目标 ID 列表中
            if current_global_idx in target_ids:
                # 命中：保持原样
                new_lines.append(line)
            else:
                # 未命中：根据功能选择处理方式
                if args.mode == 2:
                    # 功能 2：将第一列置为 1
                    parts = line.split()
                    if parts:
                        parts[0] = "1"
                        new_lines.append(" ".join(parts))
                # 功能 1 时，直接跳过不添加，即“仅保留”

            current_global_idx += 1

        if len(new_lines) > 0:
            out_path = output_dir / txt_path.name
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(new_lines) + "\n")
            out_count += 1

    print(f"[INFO] Success! Generated {out_count} files in {output_dir.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter label files based on Global ID")
    parser.add_argument("--input", type=str, required=True, help="Input folder with .txt labels")
    parser.add_argument("--ids", type=str, required=True, help="Text file with global IDs (one per line)")
    parser.add_argument("--lbl-idx-offset", type=int, default=0, help="Offset adding to labels global index  (default: 0)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output folder")
    parser.add_argument("-m", "--mode", type=int, choices=[1, 2], default=1,
                        help="Mode 1: Keep only matched IDs. Mode 2: Keep all, set unmatched first column to 1")

    args = parser.parse_args()
    run_prediction = run_filter(args)
