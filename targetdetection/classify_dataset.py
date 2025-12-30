#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据txt文件列表筛选文件并分类移动
"""

import argparse
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def move_file(args):
    """移动单个文件的函数，用于多进程"""
    src_path, dst_path = args
    try:
        shutil.move(str(src_path), str(dst_path))
        return True
    except Exception as e:
        print(f"移动文件失败 {src_path}: {e}")
        return False


def filter_and_move_files(input_dir: Path,list_file: Path, target_dir: Path, other_dir: Path, suffix: str = "", num_workers: int = None):
    """
    根据txt文件列表筛选并移动文件

    Args:
        input_dir: 输入目录
        list_file: 包含文件名的txt文件
        target_dir: 匹配文件的目标目录
        other_dir: 不匹配文件的目标目录
        suffix: 文件名后缀（在读取txt时忽略的部分）
        num_workers: 进程数
    """
    # 读取txt文件，忽略第一行的"NAME"
    with open(list_file, 'r') as f:
        lines = f.readlines()

    # 跳过第一行"NAME"，处理后续行
    target_names = set()
    for line in lines[1:]:
        name = line.strip()
        # 移除指定的后缀（如_bin2）
        if suffix and name.endswith(suffix):
            name = name[:-len(suffix)]
        target_names.add(name)
    print(f"[INFO] 从列表文件中读取到 {len(target_names)} 个目标文件名")

    # 获取所有文件
    all_files = [f for f in input_dir.iterdir() if f.is_file()]
    if not all_files:
        print(f"[WARNING] 目录 {input_dir} 中没有文件")
        return
    print(f"[INFO] 找到 {len(all_files)} 个文件")

    # 准备移动任务
    move_tasks = []
    match_count = 0
    other_count = 0

    for file_path in all_files:
        # 获取文件名（不含扩展名）
        file_stem = file_path.stem
        # 如果文件名在目标集合中
        if file_stem in target_names:
            dst_path = target_dir / file_path.name
            move_tasks.append((file_path, dst_path))
            match_count += 1
        else:
            dst_path = other_dir / file_path.name
            move_tasks.append((file_path, dst_path))
            other_count += 1
    print(f"[INFO] 匹配文件: {match_count} 个")
    print(f"[INFO] 其他文件: {other_count} 个")

    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)
    # 使用多进程移动文件
    success_count = 0
    failed_count = 0
    if num_workers is None:
        num_workers = min(os.cpu_count(), len(move_tasks))
    print(f"[INFO] 使用 {num_workers} 个进程移动文件...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(move_file, task) for task in move_tasks]
        with tqdm(total=len(futures), desc="移动文件", unit="file") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                else:
                    failed_count += 1
                pbar.update(1)

    print(f"\n[INFO] 完成!")
    print(f"  成功: {success_count} 个文件")
    print(f"  失败: {failed_count} 个文件")
    print(f"  目录结构:")
    print(f"    匹配文件: {target_dir} ({len(list(target_dir.iterdir()))} 个文件)")
    print(f"    其他文件: {other_dir} ({len(list(other_dir.iterdir()))} 个文件)")


def main():
    parser = argparse.ArgumentParser(description="根据txt文件列表筛选并移动文件")
    parser.add_argument("--input_dir", "-i", type=str, help="输入目录路径")
    parser.add_argument("--list_file", "-l", type=str, help="包含文件名的txt文件路径")
    parser.add_argument("--target-dir", "-t", type=str, default="bad", help="匹配文件的目标目录（默认: input_dir/bad）")
    parser.add_argument("--other-dir", "-o", type=str, default="good", help="不匹配文件的目标目录（默认: input_dir/good）")
    parser.add_argument("--suffix", "-s", type=str, default="_bin2", help="从txt文件名中忽略的后缀（默认: _bin2）")
    parser.add_argument("--workers", "-w", type=int, default=None, help="进程数，默认使用CPU核心数")
    args = parser.parse_args()

    # 检查路径
    input_path = Path(args.input_dir)
    list_path = Path(args.list_file)
    target_path = input_path / Path(args.target_dir)
    other_path = input_path / Path(args.other_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"[ERROR] 输入目录不存在或不是目录: {input_path}")
        exit(0)
    if not list_path.exists():
        print(f"[ERROR] 列表文件不存在: {list_path}")
        exit(0)

    try:
        filter_and_move_files(input_dir=input_path, list_file=list_path, target_dir=target_path, other_dir=other_path, suffix=args.suffix, num_workers=args.workers)
    except Exception as e:
        print(f"[ERROR] 执行失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()