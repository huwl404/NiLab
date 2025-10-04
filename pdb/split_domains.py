#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : split_domains.py
# Time       ：2025/10/4 13:11
# Author     ：Jago
# Email      ：huwl@hku.hk
# Description：
Split a protein chain into segments (domains or arbitrary ranges) and arrange each segment as a separate chain along the x-axis for visualization purposes.
The relative coordinates within each segment are preserved, but the segments are translated (not rotated).
Chemical validity is not maintained.
"""
import argparse
import sys
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from collections import OrderedDict
import string


# ---------------- utility ----------------
def parse_domains(domstr):
    """Parse domain range string, e.g. '10-80,130-142,150-285' -> [(10,80), (130,142), (150,285)]"""
    parts = [p.strip() for p in domstr.split(",") if p.strip()]
    domains = []
    for p in parts:
        if "-" not in p:
            print(f"[Skipped] Wrong domain {p}")
        s, e = p.split("-", 1)
        domains.append((int(s), int(e)))
    return domains


def make_chain_id(i):
    """Generate unique chain IDs: A..Z, a..z, 0..9, A1, A2, ..."""
    chars = list(string.ascii_uppercase) + list(string.ascii_lowercase) + list(string.digits)
    if i < len(chars):
        return chars[i]
    a = i // len(chars)
    b = i % len(chars)
    return f"{chars[a]}{chars[b]}"


def build_segments(chain, dom_ranges, split_by):
    """Build residue ranges for splitting based on domain boundaries."""
    resseqs = sorted({r.id[1] for r in chain})
    if not resseqs:
        raise ValueError("The chain contains no residues.")
    min_r, max_r = resseqs[0], resseqs[-1]
    segments = []

    if split_by == "start":
        cuts = [s for (s, e) in dom_ranges]
        # remove duplicate and then sort
        cuts = sorted(set(cuts))
        # Add N-terminal segment if any residues before first cut
        if min_r < cuts[0]:
            segments.append((min_r, cuts[0] - 1))
        for i in range(len(cuts)):
            s = cuts[i]
            e = cuts[i + 1] - 1 if i + 1 < len(cuts) else max_r
            segments.append((s, e))
        return segments
    else:  # split_by == "end"
        cuts = [e for (s, e) in dom_ranges]
        cuts = sorted(set(cuts))
        # First segment from N-terminus to first cut
        segments.append((min_r, cuts[0], None))
        for i in range(1, len(cuts)):
            segments.append((cuts[i - 1] + 1, cuts[i]))
        # Add C-terminal segment if residues after last cut
        if cuts[-1] < max_r:
            segments.append((cuts[-1] + 1, max_r))
        return segments


def rotation_matrix_from_vectors(a, b):
    """
    Return rotation matrix that rotates vector a to vector b.
    Both a and b are 3-element numpy arrays.
    If a and b are parallel, returns identity or 180-degree rotation accordingly.
    Uses Rodrigues' rotation formula.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.allclose(v, 0):
        # vectors are parallel or anti-parallel
        if c > 0.999999:
            return np.eye(3)  # same direction
        else:
            # opposite direction: rotate 180 degrees around any orthogonal axis
            # find a vector orthogonal to a
            orth = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(orth, a)) > 0.9:
                orth = np.array([0.0, 1.0, 0.0])
            axis = np.cross(a, orth)
            axis = axis / np.linalg.norm(axis)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            return np.eye(3) + 2.0 * K @ K  # rotation by pi: R = I + 2 K^2
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + (kmat @ kmat) * ((1 - c) / (s**2))
    return R


def create_translated_structure(orig_chain, segments, baseline_yz, gap):
    """
    Translate and rotate each segment along the x-axis and output a new structure.
    Each segment becomes a new chain. The internal coordinates are preserved by applying
    a rigid rotation (so that segment N->C aligns with +x) and then translating the N to desired x.
    """
    new_struct = Structure("S")
    new_model = Model(0)
    new_struct.add(new_model)

    cur_x = 0.0
    prev_right = cur_x - gap  # so desired_left for first segment is cur_x
    baseline_y, baseline_z = float(baseline_yz[0]), float(baseline_yz[1])
    chain_index = 0
    for seg in segments:
        s, e = seg
        residues = [r for r in orig_chain if s <= r.id[1] <= e]
        if not residues:
            print(f"[Skipped] segment [{s}-{e}] empty.")
            continue

        ca_residues = [r for r in residues if 'CA' in r]
        n = len(ca_residues)
        if n == 0:
            print(f"[Skipped] segment [{s}-{e}] has no CA atoms.")
            continue

        # if use_original_span:
        #     # Compute backbone path length: sum of consecutive CA-CA distances (3D)
        #     ca_coords = np.array([r['CA'].get_coord() for r in ca_residues])
        #     # sum of distances between successive CA coords
        #     diffs = ca_coords[1:] - ca_coords[:-1]
        #     seg_length = float(np.sum(np.linalg.norm(diffs, axis=1))) * binning
        # else:
        #     # Use ideal per-residue spacing along x-axis
        #     seg_length = distance * binning

        # old N and old C (first and last CA in this segment)
        old_N = ca_residues[0]['CA'].get_coord().copy()
        old_C = ca_residues[-1]['CA'].get_coord().copy()
        old_vec = old_C - old_N
        norm_old = np.linalg.norm(old_vec)

        # desired N and desired vector along +x for this segment
        desired_N = np.array([cur_x, baseline_y, baseline_z])
        desired_vec = np.array([1.0, 0.0, 0.0])

        # if old_vec is degenerate (zero or very small), skip rotation, only translate
        if norm_old <= 1e-6:
            R = np.eye(3)
        else:
            # rotation matrix to map old_vec -> desired_vec
            R = rotation_matrix_from_vectors(old_vec, desired_vec)

        # apply rigid transform: for each atom: new = R @ (coord - old_N) + desired_N
        new_chain = Chain(make_chain_id(chain_index))
        for r in residues:
            # create new residual with same id/resname/segid
            new_res = Residue(r.id, r.get_resname(), r.segid)
            for atom in r:
                old_coord = atom.get_coord()
                rotated = R.dot(old_coord - old_N) + desired_N
                new_at = Atom(
                    atom.get_name(),
                    rotated,
                    atom.get_bfactor(),
                    atom.get_occupancy(),
                    atom.get_altloc(),
                    atom.get_fullname(),
                    serial_number=atom.get_serial_number(),
                    element=getattr(atom, "element", None),
                )
                new_res.add(new_at)
            new_chain.add(new_res)

        # --- ensure this segment's leftmost atom is at prev_right + gap ---
        atom_xs = []
        for new_res in new_chain:
            atom_xs.append(new_res['CA'].get_coord()[0])
        min_x, max_x = min(atom_xs), max(atom_xs)
        # desired left position = prev_right + gap
        desired_left = prev_right + gap
        # compute additional shift to move leftmost atom to desired_left
        shift = desired_left - min_x
        if shift > 0:
            # translate all atoms in new_chain by +shift in x
            for new_res in new_chain:
                for atom in new_res:
                    c = atom.get_coord()
                    atom.set_coord(np.array([c[0] + shift, c[1], c[2]]))
        # now update prev_right and cur_x for next segment
        prev_right = max_x + shift
        cur_x = prev_right  # next desired left will be prev_right + gap
        chain_index += 1
        new_model.add(new_chain)

    return new_struct


def main():
    parser = argparse.ArgumentParser(
        description="Split a PDB chain into segments and arrange them along the x-axis as separate chains.")
    parser.add_argument("-i", "--pdb_in", required=True)
    parser.add_argument("-o", "--pdb_out", required=True)
    parser.add_argument("-c", "--chain", default="A", help="Chain ID to process (default: A).")
    parser.add_argument("-d", "--domains", required=True, help="Domain ranges, e.g. 10-80,130-142,150-285,...")
    parser.add_argument("--split_by", choices=("start", "end"), default="start",
                        help="Use the 'start' or 'end' positions of domains as split points (default: start).")
    parser.add_argument("--gap", type=float, default=10,
                        help="Distance per residue (Å) along the x-axis when computing segment length (default: 10).")
    # parser.add_argument('--use_ori_span', dest='use_ori_span', action='store_true')
    # parser.add_argument('--no_ori_span', dest='use_ori_span', action='store_false')
    args = parser.parse_args()

    dom_ranges = parse_domains(args.domains)

    p = PDBParser(QUIET=True)
    struct = p.get_structure("S", args.pdb_in)
    model = struct[0]
    if args.chain not in model:
        print(f"[Failed] Cannot find chain {args.chain}.", file=sys.stderr)
        sys.exit(1)
    chain = model[args.chain]

    segments = build_segments(chain, dom_ranges, args.split_by)
    ca_coords = []
    for r in chain:
        if 'CA' in r:
            ca_coords.append(r['CA'].get_coord()[1:3])  # y, z
    if ca_coords:
        ca_coords = np.array(ca_coords)
        baseline_yz = (float(np.mean(ca_coords[:, 0])), float(np.mean(ca_coords[:, 1])))
    else:
        print(f"[Failed] Cannot find CA atom in {args.chain}.", file=sys.stderr)
        sys.exit(1)

    new_struct = create_translated_structure(chain, segments, baseline_yz, args.gap)

    io = PDBIO()
    io.set_structure(new_struct)
    io.save(args.pdb_out)
    print(f"[Done] Written {args.pdb_out}")


if __name__ == "__main__":
    main()
