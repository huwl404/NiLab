#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : color_consurf.py
# Time       : 2025/12/30 9:45
# Author     : 14750
# Email      : huwl@hku.hk
# Description：

"""
from chimerax.core.commands import run


def color_consurf(session, chains=None, target="c"):
    """
    ChimeraX 脚本：根据 B-factor (Consurf 分数 1-9) 着色

    参数:
    - chains: 字符串或列表，例如 "A" 或 ["A", "B"]。如果为 None，则着色所有链。
    - target: 着色目标，可选 "c", "a", "s", "p" 等。
    """

    # 1. 定义 Consurf 标准颜色映射 (1-9分)
    # 格式：{B-factor分数: "十六进制颜色"}
    consurf_colors = {
        1: "#0a7c81",  # 极度变异 (Cyan)
        2: "#4bafbe",
        3: "#a5dce6",
        4: "#d7f0f0",
        5: "#FFFFFF",  # 中性 (White)
        6: "#faebf5",
        7: "#fac8dc",
        8: "#f07daa",
        9: "#a0285f",  # 极度保守 (Maroon)
        10: "#FFFF96"  # 数据不足 (Yellow)
    }

    # 2. 构建链选择字符串
    chain_sel = ""
    if chains:
        if isinstance(chains, str):
            chain_sel = f"/{chains}"
        else:
            chain_sel = f"/{','.join(chains)}"

    # 3. 初始化：先将指定范围染成灰色 (参考原脚本 cmd.color("gray", selection))
    # @@bfactor>0 选择所有具有 B-factor 属性的原子
    base_sel = f"{chain_sel}@@bfactor>0" if chain_sel else "@@bfactor>0"
    run(session, f"color {base_sel} gray target {target}")

    # 4. 循环着色
    print(f"--- Starting Consurf coloring on target: {target} ---")
    for score, color_hex in consurf_colors.items():
        # ChimeraX 选择语法说明：
        # @@bfactor=X 表示选择 B-factor 等于 X 的原子
        # 若需要支持浮点数区间，可改用 @@bfactor>=X 且 @@b<Y
        if chain_sel:
            spec = f"{chain_sel}@@bfactor={score}"
        else:
            spec = f"@@bfactor={score}"

        # 执行着色命令
        # 使用 target 参数指定着色范围（cartoon, atoms, 等）
        try:
            run(session, f"color {spec} {color_hex} target {target}")
        except Exception as e:
            # 防止某些分数在 PDB 中不存在导致报错
            continue

    # 5. 设置默认显示模式 (参考原脚本最后几行)
    if target == "cartoon":
        run(session, f"show {chain_sel if chain_sel else 'all'} cartoon")

    print("--- Coloring Complete ---")


# 在 ChimeraX 中注册该函数，使其可以通过命令行调用（可选）
# usage: color_consurf chains A target c
from chimerax.core.commands import register, StringArg, ListOf, CmdDesc

desc = CmdDesc(
    synopsis="color_consurf [chains <chains>] [target <target>]",
    keyword=[
        ("chains", ListOf(StringArg)),
        ("target", StringArg)
    ]
)
register("color_consurf", desc, color_consurf)