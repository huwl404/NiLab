#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : straighten_loops.py
# Time       ：2025/10/3 9:53
# Author     ：Jago
# Email      ：huwl@hku.hk
# Description：
deprecated
把指定链上按 domain ranges 的相邻 domain 之间的 loop 拉直（线性插值），并输出新 PDB。
此脚本会破坏化学合理性，仅作可视化。
"""
import argparse
import sys
import numpy as np
from Bio.PDB import PDBParser, PDBIO


def parse_domains(domstr):
    # domstr example: "1-200,201-420,421-650"
    parts = [p.strip() for p in domstr.split(",") if p.strip()]
    domains = []
    for p in parts:
        if "-" not in p:
            print(f"[Skipped] Wrong domain {p}")
        s, e = p.split("-", 1)
        domains.append((int(s), int(e)))
    return domains


def ca_coord_of_res(chain, seqnum):
    for r in chain:
        if r.id[1] == seqnum and 'CA' in r:
            return r['CA'].get_coord().copy()
    return None


def translate_domain_atoms(chain, start, end, vec):
    for r in chain:
        rn = r.id[1]
        if start <= rn <= end:
            for atom in r.get_atoms():
                atom.set_coord(atom.get_coord() + vec)


def main():
    parser = argparse.ArgumentParser(
        description="Straighten loops and align loops to x-axis in PDB (Only for visualization).")
    parser.add_argument("-i", "--pdb_in", required=True)
    parser.add_argument("-o", "--pdb_out", required=True)
    parser.add_argument("-c", "--chain", default="A", help="Chain ID (default: A)")
    parser.add_argument("-d", "--domains", required=True, help="domain ranges，comma split, e.g. 1-200,250-420,480-600")
    parser.add_argument("--baseline", nargs=2, type=float, metavar=("Y", "Z"),
                        help="y z values of baseline, space split（default: central line of all anchor points）")
    args = parser.parse_args()

    doms = parse_domains(args.domains)
    if len(doms) < 2:
        print("[Failed] at least 2 domains.", file=sys.stderr)
        sys.exit(1)

    print("------START------")
    p = PDBParser(QUIET=True)
    struct = p.get_structure("S", args.pdb_in)
    model = struct[0]
    if args.chain not in model:
        print(f"[Failed] Cannot find chain {args.chain}.", file=sys.stderr)
        sys.exit(1)
    chain = model[args.chain]

    # 收集每个相邻对的锚点（end_i, start_{i+1}）
    anchor_pairs = []
    for i in range(len(doms) - 1):
        end_i = doms[i][1]
        start_i1 = doms[i + 1][0]
        caA = ca_coord_of_res(chain, end_i)
        caB = ca_coord_of_res(chain, start_i1)
        if caA is None or caB is None:
            print(f"[Failed] Cannot find CA of domain {i + 1} {end_i} or domain {i + 2} {start_i1}.", file=sys.stderr)
            sys.exit(1)
        anchor_pairs.append({
            "end_res": end_i,
            "start_res": start_i1,
            "caA": caA,
            "caB": caB,
            "dist": 3.8 * (start_i1 - end_i)  # 3.8 is average distance between 2 CA
        })

    if args.baseline:
        baseline_y, baseline_z = args.baseline
    else:
        ys = []
        zs = []
        for ap in anchor_pairs:
            ys.append(ap["caA"][1])
            zs.append(ap["caA"][2])
            ys.append(ap["caB"][1])
            zs.append(ap["caB"][2])
        baseline_y = float(np.mean(ys))
        baseline_z = float(np.mean(zs))

    # 计算每个锚点的新位置
    new_positions = dict()  # key: (resnum) -> new coord (x,y,z)
    # 第一个 anchor pair 的 end 的 x 作为起点
    cur_x = float(anchor_pairs[0]["caA"][0])
    for ap in anchor_pairs:
        new_positions['end', ap["end_res"]] = np.array([cur_x, baseline_y, baseline_z])
        next_x = cur_x + ap["dist"]
        new_positions['start', ap["start_res"]] = np.array([next_x, baseline_y, baseline_z])
        cur_x = next_x

    # 现在对每个 domain 做整体平移
    first_start, last_end = doms[0][0], doms[-1][1]
    for dom in doms:
        s, e = dom
        new_pos = new_positions.get(('end', e))
        old_ca = ca_coord_of_res(chain, e)
        if s == first_start:  # for the N-terminus
            trans = new_pos - old_ca
            translate_domain_atoms(chain, 1, s - 1, trans)
        elif e == last_end:  # for the last domain and C-terminus
            new_pos = new_positions.get(('start', s))
            old_ca = ca_coord_of_res(chain, s)
            trans = new_pos - old_ca
            translate_domain_atoms(chain, s, len(chain), trans)
        else:
            trans = new_pos - old_ca
            translate_domain_atoms(chain, s, e, trans)

    # 处理每个 loop（end_i+1 ... start_{i+1}-1）
    for ap in anchor_pairs:
        s, e = ap["start_res"], ap["end_res"]
        new_start = new_positions.get(('start', s))
        new_end = new_positions.get(('end', e))
        loop_resnums = [r.id[1] for r in chain if e < r.id[1] < s]
        n = len(loop_resnums)
        for idx, resnum in enumerate(loop_resnums, start=1):
            frac = idx / (n + 1)
            new_ca = new_end + frac * (new_start - new_end)
            old_ca = ca_coord_of_res(chain, resnum)
            trans = new_ca - old_ca
            for r in chain:
                if r.id[1] == resnum:
                    res = r
            for atom in res.get_atoms():
                atom.set_coord(atom.get_coord() + trans)

    io = PDBIO()
    io.set_structure(struct)
    io.save(args.pdb_out)
    print(f"Written {args.pdb_out}")
    print("------DONE------")


if __name__ == "__main__":
    main()
