#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : show_labels.py
# Time       ：2025/10/20 19:16
# Author     ：Jago
# Email      ：huwl@hku.hk
# Description：
Show and inspect particle annotations by overlaying YOLO-style label boxes on corresponding images.
For each matching <name>.mrc/<name>.tif and <name>.txt pair, the script:
  1. Reads the image and parses the label file containing class x_center y_center width height.
  2. Converts normalized YOLO coordinates into pixel positions based on the image width and height.
  3. Displays the image with red bounding boxes and yellow center markers for each labeled particle.
  4. Allows interactive navigation:
    n / → — next image
    p / ← — previous image
    s — save current overlay as PNG in <out>/ (default overlays/)
    q / Esc — quit viewer
Examples:
View overlays for a single dataset containing .tif images:
    python show_labels.py -m ./output/map -l ./output/label
"""
import argparse
import os
import sys

import cv2
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple


def find_pairs(map_dir: Path, label_dir: Path, map_ext: str = ".mrc", label_ext: str = ".txt"):
    maps = sorted(map_dir.glob(f"*{map_ext}"))
    pairs = []
    for m in maps:
        name = m.stem
        lab = label_dir / (name + label_ext)
        if lab.exists():
            pairs.append((m, lab))
    return pairs


def read_labels(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """return list of tuples: (class, x_center, y_center, width, height)"""
    pts = []
    with txt_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            pts.append((cls, x_c, y_c, w, h))
    return pts


class Viewer:
    def __init__(self, pairs, overlays_out: Path):
        self.pairs = pairs
        self.index = 0
        self.overlays_out = overlays_out
        self.fig, self.ax = plt.subplots()
        self.im = None
        self.rects = []  # 存当前图片的补丁
        self.scatter = None
        self._cid = self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def show_current(self):
        self.ax.clear()
        mrc_path, txt_path = self.pairs[self.index]
        map_ext = os.path.splitext(mrc_path)[-1]
        if map_ext in [".mrc", ".map"]:
            m = mrcfile.open(mrc_path, permissive=True)  # shape = (y, x)
            img = np.asarray(m.data).astype(np.int16)  # int16
            m.close()
        elif map_ext in [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"]:
            # unless you assign integer unchanged, otherwise it would transform int16 to int8.
            img = cv2.imread(mrc_path, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError(f"Unsupported file type: {map_ext}")

        h, w = img.shape[0], img.shape[1]
        self.ax.set_title(f"{mrc_path.name}")
        self.im = self.ax.imshow(img, cmap="gray", origin="upper")
        labels = read_labels(txt_path)
        print(f"{mrc_path} has {len(labels)} points.")
        self.rects = []
        xs = []
        ys = []
        # draw rectangles
        for (cls, x_c, y_c, bw, bh) in labels:
            xc = x_c * w
            yc = y_c * h
            bw_px = bw * w
            bh_px = bh * h
            x0 = xc - bw_px / 2.0
            y0 = yc - bh_px / 2.0
            rect = patches.Rectangle((x0, y0), bw_px, bh_px, linewidth=1.0, edgecolor="red", facecolor="none")
            self.ax.add_patch(rect)
            self.rects.append(rect)
            xs.append(xc)
            ys.append(yc)
        # draw center points
        if xs and ys:
            self.scatter = self.ax.scatter(xs, ys, s=10, c="yellow", marker="x")
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        key = event.key
        if key is None:
            return
        if key in ("n", "right"):
            self.next()
        elif key in ("p", "left"):
            self.prev()
        elif key == "s":
            self.save_overlay()
        elif key in ("q", "escape"):
            plt.close(self.fig)

    def next(self):
        if self.index < len(self.pairs) - 1:
            self.index += 1
        else:
            self.index = 0
        self.show_current()

    def prev(self):
        if self.index > 0:
            self.index -= 1
        else:
            self.index = len(self.pairs) - 1
        self.show_current()

    def save_overlay(self):
        mrc_path, txt_path = self.pairs[self.index]
        name = mrc_path.stem + "_overlay.png"
        outp = self.overlays_out / name
        self.overlays_out.mkdir(parents=True, exist_ok=True)
        try:
            self.fig.savefig(outp, dpi=150, bbox_inches="tight")
            print(f"[INFO] Saved overlay: {outp}")
        except Exception as e:
            print(f"[ERROR] Save overlay failed: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Show particles from label txt on corresponding mrc images.")
    ap.add_argument("--images", "-m", required=True, help="folder with image files")
    ap.add_argument("--labels", "-l", required=True, help="folder with label .txt files (same basename)")
    ap.add_argument("--map-ext", default=".png", help="extension for map files (default: .png)")
    ap.add_argument("--txt-ext", default=".txt", help="extension for label files (default: .txt)")
    ap.add_argument("--start", type=int, default=0, help="start index (0-based)")
    ap.add_argument("--out", default="overlays", help="folder to save overlay PNGs (default: overlays/)")
    args = ap.parse_args()

    map_dir = Path(args.images)
    label_dir = Path(args.labels)
    if not map_dir.exists() or not map_dir.is_dir():
        print("maps folder not found", file=sys.stderr)
        sys.exit(2)
    if not label_dir.exists() or not label_dir.is_dir():
        print("labels folder not found", file=sys.stderr)
        sys.exit(2)

    pairs = find_pairs(map_dir, label_dir, map_ext=args.map_ext, label_ext=args.txt_ext)
    if not pairs:
        print("No matching .mrc/.txt pairs found.", file=sys.stderr)
        sys.exit(1)

    pairs = pairs[args.start:] + pairs[:args.start]  # rotate start
    overlays_out = Path(args.out)
    viewer = Viewer(pairs, overlays_out)
    viewer.show_current()
    plt.show()


if __name__ == "__main__":
    main()
