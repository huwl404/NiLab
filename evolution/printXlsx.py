#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : printXlsx.py
# Time       ：2025/6/29 13:08
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
"""
import pandas as pd
from pathlib import Path

# ==== 参数部分 ====
file1_path = r'D:\CourseStudy\0research_project\RUN\TN\DRE project\Nido_ORF1_info_annotated.xlsx'        # 文件1路径
file2_path = r'D:\CourseStudy\0research_project\RUN\TN\DRE project\LocalFoldSeek_aln_rmsd.xlsx'          # 文件2路径
sheet1_name = 'Nido_ORF1_info_annotated'          # 文件1的sheet名
sheet2_name = 'SARS-CoV-2-Y1-domain_aln'          # 文件2的sheet名
column_file1_path = 4           # 文件1中cif路径所在列（0起始）
column_file2_filename = 1       # 文件2中cif文件名列（第2列索引1）
column_file2_col9 = 8           # 文件2第9列索引
column_file2_col10 = 9          # 文件2第10列索引
# ===================

# 加载两个Excel文件
df1 = pd.read_excel(file1_path, sheet_name=sheet1_name)
df2 = pd.read_excel(file2_path, sheet_name=sheet2_name)

# 预处理文件2：建立文件名到第9列和第10列信息的映射
file2_dict = {}
for _, row in df2.iterrows():
    filename = str(row[column_file2_filename]).strip()
    col9 = str(row[column_file2_col9]).strip()
    col10 = str(row[column_file2_col10]).strip()
    file2_dict[filename] = f"{col9}-{col10}"

# 遍历文件1
for _, row in df1.iterrows():
    path_value = row[column_file1_path]
    if pd.isna(path_value):
        continue  # 跳过空路径

    file_path = Path(str(path_value).strip())
    file_name = file_path.stem

    if file_name in file2_dict:
        print(f"{file_name}\t{file2_dict[file_name]}")
    else:
        print(f"{file_name}")
