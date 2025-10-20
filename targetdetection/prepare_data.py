#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : prepare_data.py
# Time       ：2025/10/18 12:52
# Author     ：Jago
# Email      ：huwl@hku.hk
# Description：:
Generate per-tile MRC maps and YOLO-style label files from a SerialEM .nav file referencing montage maps.
For each .nav file, the script:
  1. Reads map and point items using utils.reader.read_nav_file().
  2. Locates referenced montage MRC files.
  3. Splits each montage into single-slice tiles according to MapFramesXY, writing them under <out>/map/ as <mapstem>_tileNNN.mrc.
  4. Collects point coordinates (XYinPc) for each tile, skips the first point per DrawnID,
  and converts pixel positions to normalized YOLO coordinates using --boxsize.
  5. Writes per-tile label files under <out>/label/ with format: class x_center y_center width height (class fixed to 0).
Examples:
Process a single .nav file and generate map/label pairs:
    python prepare_data.py -i /path/to/nav001.nav -o ./output -b 150
Re-run and overwrite existing files:
    python prepare_data.py -i /path/to/session.nav -o ./output -b 150 --override
"""
import argparse
import sys
import mrcfile
from pathlib import Path
from typing import Tuple, List, Dict

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


def process_nav(nav_path: Path, out_dir: Path, boxsize: int, overwrite: bool = False):
    print(f"[INFO] Reading nav: {nav_path}")
    items = reader.read_nav_file(str(nav_path))
    map_items, point_items = find_map_and_point_items(items)
    print(f"[INFO] Found {len(map_items)} map(s), {len(point_items)} point(s).")

    maps = build_maps_dict(nav_path, map_items)
    if not maps:
        print("[ERROR] No usable map items found. Exiting.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    map_out = out_dir / "map"
    map_out.mkdir(parents=True, exist_ok=True)
    label_out = out_dir / "label"
    label_out.mkdir(parents=True, exist_ok=True)
    summary = {"maps_processed": 0, "tiles_written": 0, "txt_written": 0}

    # split montage into tiles
    tile_info_by_map: Dict[int, Dict[int, Dict]] = {}
    for map_id, md in maps.items():
        mpath = md["MapPath"]
        nx, ny = md["MapFramesXY"]
        total_tiles = nx * ny
        # Open the file in memory-mapped mode
        mrc = mrcfile.mmap(mpath, mode='r+')
        # in imod, mrc.data is in x, y, z (col, row, sec); but in mrcfile, mrc.data is in z, y, x!!!
        if mrc.data.shape[0] != total_tiles:
            print(f"[Error] Montage tiles do not match with MapFramesXY. Skipped.")
            continue

        x_len, y_len = mrc.data.shape[2], mrc.data.shape[1]
        tile_info = {}
        for piece in range(total_tiles):
            tile_name = f"{mpath.stem}_tile{piece:03d}"
            tile_path = map_out / (tile_name + ".mrc")
            txt_path = label_out / (tile_name + ".txt")
            tile_info[piece] = {
                "tile_path": tile_path,
                "txt_path": txt_path,
                "x_len": x_len,
                "y_len": y_len
            }
            if tile_path.exists() and not overwrite:
                print(f"[INFO] Tile exists: {tile_path}. Skipped.")
                continue

            try:
                tile = mrcfile.new_mmap(tile_path, shape=(y_len, x_len), mrc_mode=1, overwrite=True)  # int16
                tile.data[:, :] = mrc.data[piece, :, :]
                summary["tiles_written"] += 1
            except Exception as e:
                print(f"[ERROR] Saving tile {tile_path}: {e}", file=sys.stderr)
        mrc.close()
        tile_info_by_map[map_id] = tile_info
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
    ap.add_argument("-o", "--out", required=True, help="output folder, folder map/ and label/ would be created under this")
    ap.add_argument("-b", "--boxsize", type=int, default=150, help="target box size in pixels on the maps")
    ap.add_argument("--override", action="store_true", help="override existing files")
    args = ap.parse_args()

    nav_path = Path(args.nav)
    if not nav_path.exists():
        print(f"[ERROR] nav file not found: {nav_path}", file=sys.stderr)
        sys.exit(2)
    out_dir = Path(args.out)
    rc = process_nav(nav_path, out_dir, args.boxsize, args.override)
    sys.exit(rc)


if __name__ == "__main__":
    main()
