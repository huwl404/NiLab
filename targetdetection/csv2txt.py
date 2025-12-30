#!/usr/bin/env python
# -*- coding:utf-8 -*-

#!/usr/bin/env python
# -*- coding:utf-8 -*-

import csv
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

def run_filter(args):
    if args.threshold < 0.5:
        print(f"[ERROR] Threshold {args.threshold} must be >= 0.5.")
        return

    csv_path = Path(args.in_csv)
    output_dir = Path(args.out_dir)
    output_txt = Path(args.out_txt)

    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {args.in_csv}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    matched_rows = []
    print(f"[INFO] Reading {args.in_csv}...")

    with open(csv_path, mode='r', encoding='utf-8') as f:
        # 使用 DictReader 方便通过列名访问数据
        reader = csv.DictReader(f)
        for row in reader:
            try:
                conf = float(row['confidence'])
                cls = row['class']
                # 筛选条件：类别匹配 且 概率 > 阈值
                if cls == args.metrics and conf > args.threshold:
                    matched_rows.append(row)
            except (ValueError, KeyError) as e:
                print(f"[WARNING] Skipping invalid row: {row}. Error: {e}")

    if not matched_rows:
        print(f"[INFO] No images match the criteria: class='{args.metrics}', threshold > {args.threshold}")
        return
    print(f"[INFO] Found {len(matched_rows)} matches. Starting to move files...")

    count = 0
    with open(output_txt, mode='w', encoding='utf-8') as txt_f:
        for item in tqdm(matched_rows, desc="Processing", unit="file"):
            src_path = Path(item['filepath'])
            if src_path.exists():
                dst_path = output_dir / src_path.name
                try:
                    # 移动文件 (如果目标已存在会覆盖)
                    shutil.move(str(src_path), str(dst_path))
                    # 写入文件名（无后缀）到 TXT
                    txt_f.write(src_path.stem + "\n")
                    count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to move {src_path.name}: {e}")
            else:
                print(f"[WARNING] File not found: {src_path}")

    print(f"[INFO] Success! List saved in: {output_txt.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter YOLO results and move files")
    parser.add_argument("-i", "--in_csv", type=str, required=True, help="input csv file")
    parser.add_argument("-m", "--metrics", type=str, required=True, help="target class name to filter (e.g. good)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="confidence threshold (default: 0.5, must >= 0.5)")
    parser.add_argument("-d", "--out_dir", type=str, default="./filtered_images", help="folder to move images into (default: ./filtered_images)")
    parser.add_argument("-o", "--out_txt", type=str, default="./filtered_images/selected_list.txt", help="output txt file for filenames (default: ./filtered_images/selected_list.txt)")

    args = parser.parse_args()
    run_filter(args)
