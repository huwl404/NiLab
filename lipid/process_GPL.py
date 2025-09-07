#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : process_GPL.py
# Time       ：2025/9/2 12:16
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
"""
import pandas as pd
import argparse


def process_lipid_file(input_path, output_path):
    # 读取Excel文件
    df = pd.read_excel(input_path)

    # 检查关键列是否存在
    required_cols = ["lipid_InChIKey", "complex_PDB_ID", "complex_Resolution"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入文件缺少必要字段: {col}")

    # 1. 去除相同 InChIKey + PDB ID 的重复项（保留第一行即可）
    df_unique = df.drop_duplicates(subset=["lipid_InChIKey", "complex_PDB_ID"], keep="first")

    # 2. 对每个 lipid_InChIKey 内部，按照 Resolution 升序排列
    df_sorted = df_unique.sort_values(by=["lipid_InChIKey", "complex_Resolution"], ascending=[True, True])

    # 3. 写出结果
    df_sorted.to_excel(output_path, index=False)
    print(f"处理完成，结果已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据 InChIKey 去重并排序的脂质数据处理脚本")
    parser.add_argument("-i", "--input", required=True, help="输入xlsx文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出xlsx文件路径")
    args = parser.parse_args()

    process_lipid_file(args.input, args.output)
