#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : trim2align.py
# Time       ：2025/6/27 20:27
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
该脚本用于批量处理结构比对并计算 RMSD，适用于结构生物学中基于模型文件（PDB/CIF）的结构片段截取与比对分析：
输入：读取包含多个 sheet 的 xlsx 文件（每行定义一个结构比对任务），根据用户指定的列索引提取 query 与 target 结构文件名及其截取范围；
自动截取结构片段：支持自动识别 query/target 模型中首个残基编号，并将指定范围偏移至实际残基编号进行截断，保存为新的 PDB 文件；
结构对齐并计算 RMSD：调用 PyMOL （通过 pymol.cmd）加载两个截断后的结构并执行 align 对齐，提取 RMSD 作为结构相似度指标；
输出：将所有处理结果写入新的 xlsx 文件，其中每个 sheet 保留原始信息并追加对应 RMSD 值。
"""

import os
import pymol
from pymol import cmd
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select


class ResidueSelect(Select):
    def __init__(self, start, end, chain_id=None):
        self.start = start
        self.end = end
        self.chain_id = chain_id

    def accept_residue(self, residue):
        res_id = residue.get_id()[1]
        if self.chain_id:
            return residue.get_parent().id == self.chain_id and self.start <= res_id <= self.end
        return self.start <= res_id <= self.end


def truncate_structure_auto_offset(input_path, output_path, start_res, end_res):
    # 解析PDB或CIF格式
    if input_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("struct", input_path)
    model = next(structure.get_models())
    chain = next(model.get_chains())
    residues = list(chain.get_residues())

    if not residues:
        raise ValueError(f"No residues found in {input_path}")

    first_residue_number = residues[0].get_id()[1]
    adjusted_start = first_residue_number + (start_res - 1)
    adjusted_end = first_residue_number + (end_res - 1)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, ResidueSelect(adjusted_start, adjusted_end, chain.id))


def calculate_rmsd_pymol_subprocess(query_path, target_path):
    # 调用 PyMOL（无界面模式 -cq） 可能由于PyMOL.exe需要使用其Python27
    # 因此该命令在cmd中正常输出结果，但在该脚本中始终报错ImportError: No module named site
    pymol.finish_launching(["pymol", "-cq"])
    try:
        # 加载两个结构
        cmd.load(query_path, "query")
        cmd.load(target_path, "target")

        # 对齐 target 到 query
        alignment_result = cmd.align("target", "query")  # [rmsd, aligned_atoms, ...]
        rmsd = alignment_result[0]

        return rmsd

    finally:
        cmd.delete("all")  # 清理工作区


def process_excel(
        xlsx_path,
        query_dir,
        target_dir,
        query_col,
        target_col,
        query_start_col,
        query_end_col,
        target_start_col,
        target_end_col,
        output_dir,
        output_xlsx_path
):
    writer = pd.ExcelWriter(output_xlsx_path, engine='openpyxl')
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # header=None, or it cannot be read correctly.
    sheets = pd.read_excel(xlsx_path, sheet_name=None, header=None, engine='openpyxl')
    for sheet_name, df in sheets.items():
        df = df.copy()
        rmsd_list = []

        for idx, row in df.iterrows():
            query_name = str(row[query_col])
            target_name = str(row[target_col])
            query_start, query_end = int(row[query_start_col]), int(row[query_end_col])
            target_start, target_end = int(row[target_start_col]), int(row[target_end_col])

            # Query and Target file path
            for ext in ['.pdb', '.cif']:
                query_path = os.path.join(query_dir, query_name + ext)
                if os.path.exists(query_path):
                    break
            else:
                rmsd_list.append("Query not found")
                continue

            for ext in ['.pdb', '.cif']:
                target_path = os.path.join(target_dir, target_name + ext)
                if os.path.exists(target_path):
                    break
            else:
                rmsd_list.append("Target not found")
                continue

            # Output filenames
            query_output = os.path.join(output_dir, f"{query_name}_{query_start}-{query_end}.pdb")
            target_output = os.path.join(output_dir, f"{target_name}_{target_start}-{target_end}.pdb")
            # Truncate structures
            truncate_structure_auto_offset(query_path, query_output, query_start, query_end)
            truncate_structure_auto_offset(target_path, target_output, target_start, target_end)

            rmsd = calculate_rmsd_pymol_subprocess(query_output, target_output)
            rmsd_list.append(rmsd)

        df["RMSD"] = rmsd_list
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.close()
    print(f"Completed. Output saved to {output_xlsx_path}")


if __name__ == "__main__":
    # col index stars from 0
    process_excel(
        xlsx_path="D:/CourseStudy/0research_project/RUN/TN/DRE project/LocalFoldSeek_aln.xlsx",
        query_dir="D:/CourseStudy/0research_project/RUN/TN/DRE project",
        target_dir="D:/CourseStudy/0research_project/RUN/TN/DRE project/Nido_ORF1_Dataset",
        query_col=0,
        target_col=1,
        query_start_col=6,
        query_end_col=7,
        target_start_col=8,
        target_end_col=9,
        output_dir="D:/CourseStudy/0research_project/RUN/TN/DRE project/LocalFoldSeek_aln_structures",
        output_xlsx_path="D:/CourseStudy/0research_project/RUN/TN/DRE project/LocalFoldSeek_aln_rmsd.xlsx"
    )
