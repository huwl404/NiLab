#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
example:
using cpu:
    python point-classifer/png2csv.py -i 20250913_eav_dmv_mc/test/good/ -o predict_good_cpu.csv
using gpu:
    python point-classifer/png2csv.py -i 20250913_eav_dmv_mc/test/bad/ -o predict_bad_gpu.csv --device 0
"""

import csv
import argparse
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

# 获取当前脚本文件所在的绝对路径
SCRIPT_DIR = Path(__file__).parent.resolve()

def run_prediction(args):
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Cannot find model {model_path}")
        return
    model = YOLO(str(model_path))

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"[ERROR] Cannot find folder {args.input}")
        return

    image_files = list(input_path.glob(f"*{args.suffix}"))
    if not image_files:
        print(f"[WARNING] No images with {args.suffix} in {args.input}")
        return
    print(f"[INFO] Found {len(image_files)} images. Starting inference...")

    # 打开 CSV 文件准备写入
    with open(args.output, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'class', 'confidence'])

        for img_p in tqdm(image_files, desc="Inference Progress", unit="img"):
            results = model(str(img_p), imgsz=args.imgsz, device=args.device, verbose=False)
            for r in results:
                if r is not None:
                    p = r.probs
                    cls_name = r.names[p.top1]
                    conf = float(p.top1conf)
                    filepath = Path(r.path)
                    writer.writerow([filepath, cls_name, f"{conf:.2f}"])

    print(f"[INFO] Success! Result saved in: {Path(args.output).absolute()}")


if __name__ == "__main__":
    # 动态设置默认模型路径为：脚本所在文件夹/11n_1K_classifier_best.pt
    default_model_path = SCRIPT_DIR / "11n_1K_classifier_best.pt"

    parser = argparse.ArgumentParser(description="YOLO classifier")
    parser.add_argument("-i", "--input", type=str, required=True, help="input folder")
    parser.add_argument("--suffix", type=str, default=".png", help="extension for input files (default: .png)")
    parser.add_argument("--imgsz", type=int, default=1024, help="inference image size (default: 1024)")
    parser.add_argument("--model", type=str, default=default_model_path, help="model path (default: 11n_1K_classifier_best.pt within this script folder)")
    parser.add_argument("--device", type=str, default="cpu", help="predicting device (default: cpu)")
    parser.add_argument("-o", "--output", type=str, default="predict_result.csv", help="output file (default: predict_result.csv)")

    args = parser.parse_args()
    run_prediction(args)
