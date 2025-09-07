#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : xlsx2fq_ecto.py
# Time       ：2025/7/10 20:39
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
"""
import os
import pandas as pd
from pymol import cmd
from pathlib import Path

# 初始化PyMOL
cmd.reinitialize()

# fq文件输出目录
OUTPUT_DIR = Path(r"D:\CourseStudy\0research_project\RUN\TN\DRE project")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取xlsx文件
df = pd.read_excel(r"D:\CourseStudy\0research_project\RUN\TN\DRE project\Nido_NSP34ecto_info.xlsx")

# 定义NSP3与NSP4区域对应的列名
# NSP3区域始终采用 predicted_structure1
# NSP4区域如果 predicted_structure2 为空，则使用 predicted_structure1，否则使用 predicted_structure2
REGIONS = {
    "NSP3_ecto": ("NSP3_ecto_start", "NSP3_ecto_end"),
    "NSP4_ecto": ("NSP4_ecto_start", "NSP4_ecto_end")
}

# 创建输出fq文件对象
output_files = {region: open(OUTPUT_DIR / f"{region}.fq", "w") for region in REGIONS}


def is_valid_pos(start, end):
    """检查残基范围是否合法"""
    try:
        start = int(start)
        end = int(end)
        return start <= end
    except Exception:
        return False


def extract_sequence(pdb_path, start, end):
    """提取结构文件中指定残基范围的氨基酸序列，默认针对链A"""
    obj_name = Path(pdb_path).stem.replace(".", "_")
    cmd.load(pdb_path, obj_name)

    # 选中指定残基范围（默认链A）
    selection = f"{obj_name} and chain A and resi {start}-{end}"
    sequence = ""
    try:
        sequence_raw = cmd.get_fastastr(selection)
        # 提取 FASTA 中的序列部分，不包含描述行
        sequence_lines = sequence_raw.strip().split("\n")
        sequence = ''.join(line for line in sequence_lines if not line.startswith(">"))
    finally:
        cmd.delete(obj_name)
    return sequence


# 主循环：逐行处理
for idx, row in df.iterrows():
    virus = str(row['virus_abbreviation'])

    # 获取 predicted_structure1 与 predicted_structure12（第二个结构）的路径
    structure1 = str(row['predicted_structure1']).strip() if pd.notna(row['predicted_structure1']) else ""
    structure2 = str(row['predicted_structure2']).strip() if pd.notna(row['predicted_structure2']) else ""

    any_region_written = False

    # ---------------------- NSP3 ecto ----------------------
    # NSP3始终使用 structure1
    if not structure1 or not os.path.isfile(structure1):
        print(f"[跳过] {virus} 的 NSP3_ecto: 文件不存在 -> {structure1}")
    else:
        nsp3_start = row.get("NSP3_ecto_start")
        nsp3_end = row.get("NSP3_ecto_end")
        if not is_valid_pos(nsp3_start, nsp3_end):
            print(f"[无效] {virus} 的 NSP3_ecto: 非法范围 -> {nsp3_start}-{nsp3_end}")
        else:
            nsp3_start = int(nsp3_start)
            nsp3_end = int(nsp3_end)
            try:
                sequence = extract_sequence(structure1, nsp3_start, nsp3_end)
                if sequence:
                    # 描述行包含病毒缩写与残基范围
                    header = f">{virus}|{nsp3_start}-{nsp3_end}"
                    output_files["NSP3_ecto"].write(f"{header}\n{sequence}\n\n")
                    any_region_written = True
                else:
                    print(f"[空序列] {virus} 的 NSP3_ecto 区段")
            except Exception as e:
                print(f"[错误] {virus} 的 NSP3_ecto 提取失败: {e}")

    # ---------------------- NSP4 ecto ----------------------
    # NSP4使用 structure2（如果非空），否则使用 structure1
    use_structure = structure2 if structure2 and structure2.strip() != "" else structure1
    if not use_structure or not os.path.isfile(use_structure):
        print(f"[跳过] {virus} 的 NSP4_ecto: 文件不存在 -> {use_structure}")
    else:
        nsp4_start = row.get("NSP4_ecto_start")
        nsp4_end = row.get("NSP4_ecto_end")
        if not is_valid_pos(nsp4_start, nsp4_end):
            print(f"[无效] {virus} 的 NSP4_ecto: 非法范围 -> {nsp4_start}-{nsp4_end}")
        else:
            nsp4_start = int(nsp4_start)
            nsp4_end = int(nsp4_end)
            try:
                sequence = extract_sequence(use_structure, nsp4_start, nsp4_end)
                if sequence:
                    header = f">{virus}|{nsp4_start}-{nsp4_end}"
                    output_files["NSP4_ecto"].write(f"{header}\n{sequence}\n\n")
                    any_region_written = True
                else:
                    print(f"[空序列] {virus} 的 NSP4_ecto 区段")
            except Exception as e:
                print(f"[错误] {virus} 的 NSP4_ecto 提取失败: {e}")

    # 完成当前病毒行处理后打印提示
    if any_region_written:
        print(f"[完成] {virus} 的 NSP3_ecto 和/或 NSP4_ecto 序列已提取")
    else:
        print(f"[跳过] {virus} 无任何合法区域被提取")

# 关闭所有输出文件
for f in output_files.values():
    f.close()

print(f"所有序列提取完毕，fq 文件已保存在：{OUTPUT_DIR.resolve()}")
