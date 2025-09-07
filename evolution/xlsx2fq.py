#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : xlsx2fq.py
# Time       ：2025/6/30 16:50
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：读取包含结构文件路径和残基范围的xlsx文件，提取对应序列，写入4个 .fq 文件中。
"""
import os
import pandas as pd
from pymol import cmd
from pathlib import Path

# 初始化PyMOL
cmd.reinitialize()

# fq文件输出目录
OUTPUT_DIR = Path(r"D:\CourseStudy\0research_project\RUN\TN\DRE project")  # 修改为你需要的路径
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取数据
df = pd.read_excel(r"D:\CourseStudy\0research_project\RUN\TN\DRE project\Nido_ecto-Y1_info.xlsx")

# 定义目标序列类型及对应的列名
REGIONS = {
    "Y1": ("Y1_start", "Y1_end"),
    "Y1_helix": ("Y1_helix_start", "Y1_helix_end"),
    "ecto": ("ecto_start", "ecto_end"),
    "ecto_helix": ("ecto_helix_start", "ecto_helix_end")
}

# 创建输出文件对象
output_files = {region: open(OUTPUT_DIR / f"{region}.fq", "w") for region in REGIONS}


def is_valid_pos(start, end):
    """检查起始位置是否为合法整数"""
    try:
        start = int(start)
        end = int(end)
        return start <= end
    except:
        return False


def extract_sequence(pdb_path, start, end):
    """提取结构文件中指定残基范围的序列"""
    obj_name = Path(pdb_path).stem.replace(".", "_")
    cmd.load(pdb_path, obj_name)

    # 选中指定残基范围，默认链A
    selection = f"{obj_name} and chain A and resi {start}-{end}"
    sequence = ""

    try:
        sequence_raw = cmd.get_fastastr(selection)
        sequence_lines = sequence_raw.strip().split("\n")
        sequence = ''.join(line for line in sequence_lines if not line.startswith(">"))
    finally:
        cmd.delete(obj_name)

    return sequence


# 主循环
for idx, row in df.iterrows():
    virus = str(row['virus_abbreviation'])
    pdb_path = str(row['predicted_structure'])
    pdb_filename = Path(pdb_path).name
    any_region_written = False

    if not os.path.isfile(pdb_path):
        print(f"[跳过] {virus}: 文件不存在 -> {pdb_path}")
        continue

    for region, (start_col, end_col) in REGIONS.items():
        start = row.get(start_col)
        end = row.get(end_col)

        if not is_valid_pos(start, end):
            print(f"[无效] {virus} 的 {region}: 非法范围 -> {start}-{end}")
            continue

        start = int(start)
        end = int(end)

        try:
            sequence = extract_sequence(pdb_path, start, end)
            if sequence:
                header = f">{virus}|{start}-{end}"
                output_files[region].write(f"{header}\n{sequence}\n\n")
                any_region_written = True
            else:
                print(f"[空序列] {virus} 的 {region} 区段")
        except Exception as e:
            print(f"[错误] {virus} 的 {region} 提取失败: {e}")

    if any_region_written:
        print(f"[完成] {virus} 所有可提取序列处理完毕")
    else:
        print(f"[跳过] {virus} 无任何合法区域被提取")

# 关闭所有输出文件
for f in output_files.values():
    f.close()

print(f"所有序列提取完毕，fq 文件已保存在：{OUTPUT_DIR.resolve()}")
