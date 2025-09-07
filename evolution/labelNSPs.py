#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : labelNSPs.py
# Time       ：2025/6/28 15:45
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
该脚本实现对病毒蛋白的批量注释和结构分析流程，包括以下主要步骤：
读取输入数据：从 xlsx 文件中读取病毒蛋白的基础信息，包括病毒缩写名、UniProt编号、文件路径等。
获取注释信息：如果没有提供 UniProt 注释文件路径，则通过 UniProt API 自动下载并保存为 JSON 文件，已存在文件将不重复下载。
寻找结构文件：如果结构文件路径为空，将自动在用户指定的结构目录中搜索符合命名规则的 .cif 或 .pdb 文件，优先使用 .cif 文件。
解析 UniProt 注释特征：统一提取 "features" 字段中结构域信息，包括：Transmembrane、Chain/Domain、其他类型。
识别二级结构（螺旋）：利用 PyMOL 脚本接口对结构文件进行分析，提取螺旋的起止位置。
整理与合并：将上述四类特征（TM、Chain、Others、SS）合并，保留每个蛋白的原始信息，同时对每类特征分配独立列，通过 zip_longest 保证齐头并列。
逐行写入 CSV 文件：每完成一个蛋白分析即写入输出文件。

结构文件的寻找逻辑存在bug，只会将匹配到的包括病毒缩写名的第一个路径写入输出文件，所以可能会出现重复，该bug暂未修复。
"""
import os
import json
from itertools import zip_longest

import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pymol2

MAX_RETRIES = 2
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/{}.json"


def fetch_uniprot_json(uniprot_id, save_path):
    """下载 UniProt JSON 文件"""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(UNIPROT_API_URL.format(uniprot_id), timeout=10)
            if response.status_code == 200:
                with open(save_path, 'w') as f:
                    json.dump(response.json(), f, indent=2)
                return save_path
        except Exception as e:
            print(f"[Retry {attempt + 1}] Failed fetching {uniprot_id}: {e}")
    print(f"[FAILURE] Could not fetch UniProt info for {uniprot_id} after {MAX_RETRIES} attempts.")
    return None


def locate_structure_file(base_dir, virus_abbr):
    """根据病毒缩写在目录中查找结构文件"""
    virus_tag = virus_abbr.lower().replace('-', '_')
    for ext in ['.cif', '.pdb']:
        for file in Path(base_dir).rglob(f"*{virus_tag}*{ext}"):
            return str(file.resolve())
    return None


def parse_features(json_data):
    """提取某个功能分类下的特征条目"""
    transmembrane, chain_domain, others = [], [], []
    for feature in json_data.get("features", []):
        start = feature.get("location", {}).get("start", {}).get("value", '')
        end = feature.get("location", {}).get("end", {}).get("value", '')
        tmp = {"type": feature.get("type"),
               "position": str(start) + '-' + str(end),
               "description": feature.get("description", '')}
        if tmp.get("type") == "Transmembrane":
            transmembrane.append(tmp)
        elif tmp.get("type") == "Chain" or tmp.get("type") == "Domain":
            chain_domain.append(tmp)
        else:
            others.append(tmp)
    return transmembrane, chain_domain, others


def extract_pymol_helices(structure_file):
    """使用 PyMOL 提取螺旋结构"""
    helices = []
    obj_name = "prot"
    with pymol2.PyMOL() as pymol:
        pymol.start()
        pymol.cmd.load(structure_file, obj_name)
        pymol.cmd.dss(obj_name)
        # cmd.iterate() 会对每个原子调用一次，这会导致同一个残基编号出现多次记录，尤其是带氢原子、多个模型、异构原子等；
        ss_all = []
        pymol.cmd.iterate(obj_name, "ss_map.append((resi, ss))", space={'ss_map': ss_all})
        stored = [(int(resi), ss) for resi, ss in ss_all if resi.isdigit() and ss]

        current_helix = []
        for resi, ss in stored:
            if ss == 'H':
                current_helix.append(resi)
            elif current_helix:
                helices.append((min(current_helix), max(current_helix)))
                current_helix = []
        if current_helix:
            helices.append((min(current_helix), max(current_helix)))

    return [
        {"type": "helix", "position": f"{start}-{end}"}
        for start, end in helices if start <= end
    ]


def merge_features(base_info, tm, cd, others, structure):
    """将四类特征合并为逐行格式"""
    merged = []
    first_line = True
    for tm, cd, others, structure in zip_longest(tm, cd, others, structure, fillvalue={}):
        row = {}
        if first_line:
            row = base_info.copy()
            first_line = False
        row.update({
            "Uniprot_TM": tm.get("type", ''),
            "TM_position": tm.get("position", ''),
            "TM_description": tm.get("description", ''),
            "Uniprot_Chain/Domain": cd.get("type", ''),
            "C/D_position": cd.get("position", ''),
            "C/D_description": cd.get("description", ''),
            "Uniprot_Others": others.get("type", ''),
            "Others_position": others.get("position", ''),
            "Others_description": others.get("description", ''),
            "Predicted_Secondary_Structure": structure.get("type", ''),
            "SS_position": structure.get("position", '')})
        merged.append(row)
    return merged


def main(input_excel, output_excel, annotation_dir, structure_dir):
    input_path = Path(input_excel)
    output_excel = output_excel or str(input_path.parent / f"{input_path.stem}_annotated.csv")
    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(structure_dir, exist_ok=True)

    df = pd.read_excel(input_excel)
    write_header = True

    for _, row in tqdm(df.iterrows(), total=len(df)):
        virus = row['virus_abbreviation'].strip()
        uniprot_id = row['uniprot_number'].strip()
        base_info = row.to_dict()

        # Step 1: 加载或下载 UniProt 注释文件
        json_path = row.get('uniprot_label')
        if pd.isna(json_path):
            json_file = Path(annotation_dir) / f"{uniprot_id}.json"
            if not json_file.exists():
                fetch_result = fetch_uniprot_json(uniprot_id, json_file)
                if not fetch_result:
                    print(f"[ERROR] Skipping {virus}: Failed to fetch UniProt data.")
                    continue
            json_path = str(json_file.resolve())
        base_info['uniprot_label'] = json_path

        # Step 2: 查找结构文件
        structure_path = row.get('predicted_structure')
        if pd.isna(structure_path):
            found_structure = locate_structure_file(structure_dir, virus)
            if found_structure:
                structure_path = found_structure
            else:
                print(f"[WARNING] Structure file not found for {virus}.")
                structure_path = ''
        base_info['predicted_structure'] = structure_path

        # Step 3: 解析注释特征
        tm, cd, others = [], [], []
        try:
            with open(json_path) as f:
                jdata = json.load(f)
                tm, cd, others = parse_features(jdata)
        except Exception as e:
            print(f"[ERROR] Failed parsing JSON for {virus}: {e}")

        # Step 4: 提取结构特征
        ss = []
        if structure_path and os.path.exists(structure_path):
            try:
                ss = extract_pymol_helices(structure_path)
            except Exception as e:
                print(f"[ERROR] PyMOL failed for {virus}: {e}")

        # Step 5: 写入输出文件
        rows = merge_features(base_info, tm, cd, others, ss)
        # to_excel()不支持追加模式
        pd.DataFrame(rows).to_csv(output_excel, index=False, mode='w' if write_header else 'a', header=write_header)
        write_header = False
        print(f"[SUCCESS] Processed {virus}")

    print(f"[DONE] Output written to {output_excel}")


if __name__ == "__main__":
    input_excel = r"D:\CourseStudy\0research_project\RUN\TN\DRE project\CoV_NSP4_info.xlsx"
    output_excel = r"D:\CourseStudy\0research_project\RUN\TN\DRE project\CoV_NSP4_info_annotated.csv"
    annotation_dir = r"D:\CourseStudy\0research_project\RUN\TN\DRE project\Nido_ORF1_annotation"
    structure_dir = r"D:\CourseStudy\0research_project\RUN\TN\DRE project\Nido_ORF1_Dataset"

    main(input_excel, output_excel, annotation_dir, structure_dir)
