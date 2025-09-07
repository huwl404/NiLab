#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : spiltXlsx.py
# Time       ：2025/6/28 16:17
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
"""
import pandas as pd

# 读取 Excel 文件中名为 sheet1 的工作表
df = pd.read_excel(r'D:\CourseStudy\0research_project\RUN\TN\DRE project\1.Classificaion of Nidovirales.xlsx')

# 提取所需列
virus_abbreviation = df['Abbreviation']
j_col = df['Uniprot number']

# 删除英文或中文冒号及其之后（包括换行符）的内容
# (?s) 是等价于 Python 中的 re.DOTALL，让 . 能匹配换行符。
j_col_cleaned = j_col.str.replace(r'(?s)[:：].*', '', regex=True)

# 分割为两个部分（uniprot number 和 uniprot name）
split_data = j_col_cleaned.str.split(';', n=1, expand=True)
split_data.columns = ['uniprot_number', 'uniprot_name']

# 构建新的 DataFrame
result = pd.DataFrame({
    'virus_abbreviation': virus_abbreviation,
    'uniprot_number': split_data['uniprot_number'].str.strip(),
    'uniprot_name': split_data['uniprot_name'].str.strip()
})

# 保存为新的 Excel 文件
result.to_excel(r'D:\CourseStudy\0research_project\RUN\TN\DRE project\Nido_ORF1_info.xlsx', index=False)
