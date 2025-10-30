#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : prepare_data.py
# Time       ：2025/10/18 12:52
# Author     ：Jago
# Email      ：huwl@hku.hk
# Description：:
Generate per-tile image files and YOLO-style label files from a SerialEM .nav file referencing montage maps.
For each .nav file, the script:
  1. Reads map and point items using utils.reader.read_nav_file().
  2. Locates referenced montage MRC files.
  3. Splits each montage into single-slice tiles according to MapFramesXY, writing them under <out>/images/ as <mapstem>_tileNNN.png (int8).
  4. Collects point coordinates (XYinPc) for each tile, skips the first point per DrawnID, and converts pixel positions to normalized YOLO coordinates using --boxsize.
  5. Writes per-tile label files under <out>/labels/ with format: class x_center y_center width height (class fixed to 0).
Examples:
Process a single .nav file and generate image/label pairs:
    python prepare_data.py -i /path/to/nav001.nav -o ./output -b 150
Re-run and overwrite existing files:
    python prepare_data.py -i /path/to/session.nav -o ./output -b 150 --override
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import mrcfile
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np

from utils import reader


def find_map_and_point_items(items):
    """Split pyserialem items into map_items and nav_items."""
    map_items = []
    point_items = []
    for it in items:
        type_tag = getattr(it, "Type", None)
        if type_tag == 2:
            map_items.append(it)
        elif type_tag == 0:
            point_items.append(it)
        # ignore 1 -> polygon
    return map_items, point_items


def build_maps_dict(nav_path: Path, map_items) -> Dict[int, Dict]:
    """Build maps dict keyed by MapID with 'MapPath' and 'MapFramesXY'."""
    maps = {}
    for m in map_items:
        mapid = getattr(m, "MapID", None)
        mapfile = getattr(m, "MapFile", None)
        mapframes = getattr(m, "MapFramesXY", None)

        # Linux can not recognize path like Z:\NI_lab\20250913_eav_spa\ as Windows!!!
        mappath = nav_path.parent / Path(mapfile.replace("\\", "/")).name
        if not mappath.exists():
            print(f"[INFO] Map can not be found: {mappath}. Skipped.", file=sys.stderr)
            continue

        maps[mapid] = {
            "MapPath": mappath,
            "MapFramesXY": mapframes,
        }
    return maps


def estimate_global_percentiles(mrc, per_tile_sample=2000, p_low=1, p_high=99):
    """Use global gray percentile to clip the range of each tile, avoiding too dark or too bright"""
    z_len = mrc.data.shape[0]
    samples = []
    for i in range(z_len):
        arr = mrc.data[i]
        flat = arr.ravel()
        step = max(1, flat.size // per_tile_sample)
        samples.append(flat[::step])
    samples = np.concatenate(samples)
    lo, hi = np.percentile(samples, (p_low, p_high))
    return float(lo), float(hi)


def save_tile(mapid: int, mpath: Path, out_dir, map_ext, lbl_ext, overwrite):
    """Use ProcessPool and cv2 to speed up writing tiles."""
    map_out = out_dir / "images"
    label_out = out_dir / "labels"
    mrc = mrcfile.mmap(mpath, mode='r+')
    x_len, y_len, z_len = mrc.data.shape[2], mrc.data.shape[1], mrc.data.shape[0]
    tile_info = {}
    written = 0
    for piece in range(z_len):
        tile_name = f"{mpath.stem}_tile{piece:03d}"
        # tile_path = map_out / (tile_name + ".mrc")
        # Ultralytics only support images with format
        # {'tif', 'png', 'mpo', 'pfm', 'jpeg', 'heic', 'tiff', 'bmp', 'jpg', 'dng', 'webp'}
        tile_path = map_out / (tile_name + map_ext)
        txt_path = label_out / (tile_name + lbl_ext)
        tile_info[piece] = {
            "tile_path": tile_path,
            "txt_path": txt_path,
            "x_len": x_len,
            "y_len": y_len
        }
        if tile_path.exists() and not overwrite:
            print(f"[INFO] Tile exists: {tile_path}. Skipped.")
            continue

        # try:
        #     # tile = mrcfile.new_mmap(tile_path, shape=(y_len, x_len), mrc_mode=1, overwrite=True)  # int16
        #     # tile.data[:, :] = mrc.data[piece, :, :]
        #     img = mrc.data[piece]
        #     img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        #     Image.fromarray(img_norm).save(tile_path)
        #     summary["tiles_written"] += 1
        # except Exception as e:
        #     print(f"[ERROR] Saving tile {tile_path}: {e}", file=sys.stderr)

        # Nearly all regular picture software would open pictures in int8 unless you explicitly assign int16!!!
        img = mrc.data[piece].astype(np.uint16)
        # To avoid transforming to float64 to compute img_norm
        img = img.astype(np.float16)
        # Ultralytics only accept int8 images to be trained and reasoned
        img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        # scale for the whole montage is actually a bad thing, it would cause training not smooth!!!
        # imgf = np.clip(img, lo, hi)
        # img_norm = ((imgf - lo) / (hi - lo) * 255.0).round().astype(np.uint8)
        cv2.imwrite(str(tile_path), img_norm)
        written += 1
    mrc.close()
    return mapid, tile_info, written


def process_nav(nav_path: Path, out_dir: Path, boxsize: int, map_ext: str, lbl_ext: str, overwrite: bool = False):
    print(f"[INFO] Reading nav: {nav_path}")
    items = reader.read_nav_file(str(nav_path))
    map_items, point_items = find_map_and_point_items(items)
    print(f"[INFO] Found {len(map_items)} map(s), {len(point_items)} point(s).")

    maps = build_maps_dict(nav_path, map_items)
    if not maps:
        print("[ERROR] No usable map items found. Exiting.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    map_out = out_dir / "images"
    map_out.mkdir(parents=True, exist_ok=True)
    label_out = out_dir / "labels"
    label_out.mkdir(parents=True, exist_ok=True)
    summary = {"maps_processed": 0, "tiles_written": 0, "txt_written": 0}

    # split montage into tiles
    tile_info_by_map: Dict[int, Dict[int, Dict]] = {}
    tasks = []
    for map_id, md in maps.items():
        mpath = md["MapPath"]
        nx, ny = md["MapFramesXY"]
        total_tiles = nx * ny
        # Open the file in memory-mapped mode
        mrc = mrcfile.mmap(mpath, mode='r+')
        # in imod, mrc.data is in x, y, z (col, row, sec); but in mrcfile, mrc.data is in z, y, x!!!
        z_len = mrc.data.shape[0]
        # gl_lo, gl_hi = estimate_global_percentiles(mrc)
        mrc.close()
        if z_len != total_tiles:
            print(f"[Error] Montage tiles do not match with MapFramesXY. Skipped.")
            continue

        tasks.append((map_id, mpath))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(save_tile, t[0], t[1], out_dir, map_ext, lbl_ext, overwrite) for t in tasks]
        for f in as_completed(futures):
            mid, tile_info, w = f.result()
            tile_info_by_map[mid] = tile_info
            summary["tiles_written"] += w
            summary["maps_processed"] += 1

    # allocate points lists per tile (class, x_center, y_center, width, height)
    points_per_tile: Dict[Path, List[Tuple[int, float, float, float, float]]] = {}
    skipped_first_point: set = set()  # first point is for z_focus, not DMV
    missing_maps: set = set()  # print missing maps
    for n in point_items:
        # get DrawnID (which map this point belongs to) one ID per-montage
        drawnid = getattr(n, "DrawnID", None)
        if drawnid in missing_maps:
            continue

        if drawnid not in tile_info_by_map:
            print(f"[INFO] Points with DrawnID {drawnid} are not on the split tiles. Skipped.")
            missing_maps.add(drawnid)
            continue

        groupid = getattr(n, "GroupID", None)
        # skip first point for each group
        if groupid not in skipped_first_point:
            skipped_first_point.add(groupid)
            continue

        pieceon = getattr(n, "PieceOn", None)
        xpic, ypic = getattr(n, "XYinPc", None)

        tile_info = tile_info_by_map[drawnid][pieceon]
        txt_path = tile_info["txt_path"]
        xrel = xpic / tile_info["x_len"]
        yrel = ypic / tile_info["y_len"]
        wrel = boxsize / tile_info["x_len"]
        hrel = boxsize / tile_info["y_len"]

        if txt_path not in points_per_tile:
            points_per_tile[txt_path] = []
        points_per_tile[txt_path].append((0, xrel, yrel, wrel, hrel))  # only 1 class for now

    # write txt files
    for txt_path, pts in points_per_tile.items():
        if txt_path.exists() and not overwrite:
            print(f"[INFO] TXT exists: {txt_path}. Skipped.")
            continue

        try:
            with txt_path.open("w", encoding="utf-8") as fh:
                for c, x, y, w, h in pts:
                    fh.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            summary["txt_written"] += 1
        except Exception as e:
            print(f"[ERROR] Writing txt {txt_path}: {e}", file=sys.stderr)

    print("------FINISH------")
    print(f"Maps processed: {summary['maps_processed']}; Tiles written: {summary['tiles_written']}; TXT files: {summary['txt_written']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Extract montage tiles and per-tile point lists from a SerialEM .nav file.")
    ap.add_argument("-i", "--nav", required=True, help="input .nav file (maps should be in the same folder with .nav)")
    ap.add_argument("-o", "--out", required=True, help="output folder, folder images/ and labels/ would be created under this")
    ap.add_argument("-b", "--boxsize", type=int, default=256, help="target box size in pixels on the maps (default: 256)")
    ap.add_argument("--map-ext", default=".png", help="extension for map files (default: .png)")
    ap.add_argument("--lbl-ext", default=".txt", help="extension for label files (default: .txt)")
    ap.add_argument("--override", action="store_true", help="override existing files")
    args = ap.parse_args()

    nav_path = Path(args.nav)
    if not nav_path.exists():
        print(f"[ERROR] nav file not found: {nav_path}", file=sys.stderr)
        sys.exit(2)
    out_dir = Path(args.out)
    rc = process_nav(nav_path, out_dir, args.boxsize, args.map_ext, args.lbl_ext, args.override)
    sys.exit(rc)


if __name__ == "__main__":
    main()
