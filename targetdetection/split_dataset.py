#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
随机按比例分割文件到train/val/test目录
支持多进程加速和进度条显示
"""

import argparse
import os
import random
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


def split_files(input_dir: Path, ratios: tuple, seed: int = 42, num_workers: int = None):
    """
    按比例分割文件到train/val/test目录

    Args:
        input_dir: 输入目录
        ratios: (train_ratio, val_ratio, test_ratio) 比例，总和为1
        seed: 随机种子
        num_workers: 进程数
    """
    # 获取所有文件（排除目录）
    all_files = [f for f in input_dir.iterdir() if f.is_file()]
    if not all_files:
        print(f"[WARNING] 目录 {input_dir} 中没有文件")
        return
    print(f"[INFO] 找到 {len(all_files)} 个文件")

    # 设置随机种子
    random.seed(seed)
    # 随机打乱文件列表
    random.shuffle(all_files)
    # 计算各集合文件数量
    n_total = len(all_files)
    n_train = int(n_total * ratios[0])
    n_val = int(n_total * ratios[1])
    n_test = n_total - n_train - n_val

    print(f"[INFO] 分割比例: train={ratios[0] * 100:.1f}%({n_train}), val={ratios[1] * 100:.1f}%({n_val}), test={ratios[2] * 100:.1f}%({n_test})")

    # 分配文件
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]

    # 创建目标目录
    train_dir = input_dir / "train"
    val_dir = input_dir / "val"
    test_dir = input_dir / "test"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # 准备移动任务
    move_tasks = []
    # train文件
    for file_path in train_files:
        dst_path = train_dir / file_path.name
        move_tasks.append((file_path, dst_path))
    # val文件
    for file_path in val_files:
        dst_path = val_dir / file_path.name
        move_tasks.append((file_path, dst_path))
    # test文件
    for file_path in test_files:
        dst_path = test_dir / file_path.name
        move_tasks.append((file_path, dst_path))

    # 使用多进程移动文件
    success_count = 0
    failed_count = 0

    # 确定进程数
    if num_workers is None:
        num_workers = min(os.cpu_count(), len(move_tasks))
    print(f"[INFO] 使用 {num_workers} 个进程移动文件...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(move_file, task) for task in move_tasks]
        # 使用tqdm显示进度
        with tqdm(total=len(futures), desc="移动文件", unit="file") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                else:
                    failed_count += 1
                pbar.update(1)

    # 输出结果
    print(f"\n[INFO] 完成!")
    print(f"  成功: {success_count} 个文件")
    print(f"  失败: {failed_count} 个文件")
    print(f"  目录结构:")
    print(f"    train: {train_dir} ({len(train_files)} 个文件)")
    print(f"      val: {val_dir} ({len(val_files)} 个文件)")
    print(f"     test: {test_dir} ({len(test_files)} 个文件)")


def main():
    parser = argparse.ArgumentParser(description="按比例随机分割文件到train/val/test目录", formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
示例:
  %(prog)s -i /path/to/data -r 0.7 0.2 0.1
  %(prog)s --input /path/to/data --ratios 0.8 0.1 0.1 --seed 123 --workers 48
        """
    )
    parser.add_argument("-i", "--input",type=str,help="输入目录路径")
    parser.add_argument("-r", "--ratios", type=float,nargs=3, default=[0.7, 0.2, 0.1], metavar=("TRAIN", "VAL", "TEST"), help="train/val/test比例，总和必须为1 (默认: 0.7 0.2 0.1)")
    parser.add_argument("-s", "--seed", type=int, default=42, help="随机种子 (默认: 42)")
    parser.add_argument("-w","--workers", type=int, default=None, help="进程数，默认使用CPU核心数")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 目录不存在: {input_path}")
        exit(1)

    if not input_path.is_dir():
        print(f"[ERROR] 不是目录: {input_path}")
        exit(1)

    # 检查比例
    if any(r < 0 for r in args.ratios):
        print("[ERROR] 比例不能为负数")
        exit(1)

    if abs(sum(args.ratios) - 1.0) > 1e-10:
        print(f"[ERROR] 比例总和必须为1，当前为 {sum(args.ratios)}")
        exit(1)

    try:
        split_files(input_dir=input_path, ratios=tuple(args.ratios), seed=args.seed, num_workers=args.workers)
    except Exception as e:
        print(f"[ERROR] 执行失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()