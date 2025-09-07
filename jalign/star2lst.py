#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : star2lst.py.py
# Time       ：2025/7/4 14:16
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
This script converts a RELION 3.1 .star file (containing particle metadata) into an EMAN2-compatible .lst file.
Features:
Support for RELION 3.1 star files with separate data_optics and data_particles blocks.
Correct transformation of Euler angles and particle shifts from RELION convention to EMAN2 format.
Automatically sets the output .lst filename if not provided.
Usage Examples:
    python star2lst.py particles.star
    python star2lst.py particles.star --lst converted.lst

20250808 revision: add support for star file without Origin columns.
"""
import os
import argparse
from EMAN2 import Transform
from EMAN2star import StarFile3


def relion_to_eman2(rot, tilt, psi, origin_x, origin_y, apix, boxsize):
    t = Transform({"type": "spider", "phi": rot, "theta": tilt, "psi": psi})
    rot_eman = t.get_rotation("eman")
    x = -origin_x / apix + boxsize / 2
    y = -origin_y / apix + boxsize / 2
    return rot_eman["alt"], rot_eman["az"], rot_eman["phi"], x, y


def main():
    parser = argparse.ArgumentParser(
        description="Convert RELION 3.1 star file to EMAN2 .lst file."
    )
    parser.add_argument("starfile", help="Input RELION .star file (with particles)")
    parser.add_argument("--lst", help="Output .lst file name", default=None)
    args = parser.parse_args()

    starfile = os.path.abspath(args.starfile)
    # 使用 StarFile3，可以完整读取所有 data_ 区块
    star = StarFile3(starfile)

    lstfile = args.lst or os.path.splitext(os.path.basename(starfile))[0] + ".lst"

    print(f"Reading STAR file: {starfile}")
    print(f"Generating LST file: {lstfile}")

    optics_fields = {
        "voltage": "rlnVoltage",
        "cs": "rlnSphericalAberration",
        "ampcont": "rlnAmplitudeContrast",
        "apix": "rlnImagePixelSize",
        "boxsize": "rlnImageSize"
    }

    optics_group_params = {}
    optics_block = star["optics"]

    for i in range(len(optics_block["rlnOpticsGroup"])):
        og = optics_block["rlnOpticsGroup"][i]
        optics_group_params[og] = {
            key: optics_block[field][i] for key, field in optics_fields.items()
        }

    with open(lstfile, "w") as f:
        # ...必须要注释LST文件！！！否则Jalign报错
        f.write("#LST\n")
        particles_block = star["particles"]
        for i in range(len(particles_block["rlnImageName"])):
            image_entry = particles_block["rlnImageName"][i]
            imgnum, path = image_entry.split("@")
            # 将 RELION 中的图像编号（从 1 开始）转换为 EMAN2 使用的图像编号（从 0 开始）。
            imgnum = int(imgnum) - 1

            rot = particles_block["rlnAngleRot"][i]
            tilt = particles_block["rlnAngleTilt"][i]
            psi = particles_block["rlnAnglePsi"][i]

            origin_x = 0.0
            origin_y = 0.0
            # 检查 origin 列是否存在
            if "rlnOriginXAngst" in particles_block and "rlnOriginYAngst" in particles_block:
                origin_x = particles_block["rlnOriginXAngst"][i]
                origin_y = particles_block["rlnOriginYAngst"][i]
            elif "rlnOriginX" in particles_block and "rlnOriginY" in particles_block:
                # 如果是像素单位，需要转成 Å
                origin_x = particles_block["rlnOriginX"][i] * apix
                origin_y = particles_block["rlnOriginY"][i] * apix
            else:
                if i == 0:  # 只提示一次
                    print("Warning: No Origin columns found, defaulting to (0,0)")

            og = particles_block["rlnOpticsGroup"][i]
            ogp = optics_group_params[og]
            apix = ogp["apix"]
            boxsize = ogp["boxsize"]

            # 粒子平移以前使用像素（ rlnOriginX 和 rlnOriginY ），但在 jalign 3.1 中改为埃（ rlnOriginXAngstrom 和 rlnOriginYAngstrom ）。
            alt, az, phi, x, y = relion_to_eman2(rot, tilt, psi, origin_x, origin_y, apix, boxsize)

            dfu = particles_block["rlnDefocusU"][i] / 1e4
            dfv = particles_block["rlnDefocusV"][i] / 1e4
            dfang = particles_block["rlnDefocusAngle"][i]
            defocus = (dfu + dfv) / 2
            dfdiff = abs(dfu - dfv) / 2
            if dfu > dfv:
                dfang = (dfang + 90) % 360

            voltage = ogp["voltage"]
            cs = ogp["cs"]
            # rlnAmplitudeContrast (double) : Amplitude contrast (as a fraction, i.e. 10% = 0.1)
            ampcont = ogp["ampcont"] * 100

            f.write(
                f"{imgnum}\t{path}\teuler={alt:.6f},{az:.6f},{phi:.6f}"
                f"\tcenter={x:.6f},{y:.6f}\tvoltage={voltage:.6f}\tcs={cs:.6f}"
                f"\tapix={apix:.6f}\tampcont={ampcont:.6f}\tdefocus={defocus:.6f}"
                f"\tdfdiff={dfdiff:.6f}\tdfang={dfang:.6f}\n"
            )

    print(f"Wrote {len(particles_block['rlnImageName'])} particles to {lstfile}")


if __name__ == "__main__":
    main()
