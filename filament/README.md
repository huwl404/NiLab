# 纤维起止点 → 拟合曲线 → 颗粒 STAR

本目录提供：

- **fiber_to_star.py**：根据纤维起止点 STAR 与 **.rec 体积**拟合每条纤维曲线（卷积 + 二值化 + 可调曲率曲线拟合，思路参考 [FibrilFinder, PMC8217313](https://pmc.ncbi.nlm.nih.gov/articles/PMC8217313/)），并（二选一）沿曲线等间距输出颗粒 STAR，或从 MOD 中过滤曲线附近颗粒后转 STAR。**可在 ChimeraX 中通过 `runscript` 直接调用**，也可在命令行独立运行。
- **analyze_mrc_rec.py**（可选）：分析 convmap.mrc 与 .rec 的信号关系；当前实现**仅使用 .rec** 作为输入体积。

## 数据约定

- **\*.rec**：结构生物学 cryoET 原始断层图重建，**必选**作为纤维拟合的体积输入。
- `*.mod`：挑选的颗粒坐标（IMOD 格式），可选，用于“过滤曲线外颗粒后转 STAR”。
- **\*.star**：本方案定义的**纤维起止点**标注：表头后每两行为一条纤维（第一行起始点，第二行终止点）；若总行数为奇数，最后一行仅起始点，脚本会提醒并照常处理其余成对。列为 `_rlnCoordinateX/Y/Z`、`_rlnOriginX/Y/Z`、`_rlnAngleRot/Tilt/Psi`。

## 参数与单位

- **--voxel-size**（**必选**）：体素尺寸，单位 **Å**。用于将体积与 STAR 坐标统一到同一尺度。
- **--radius**、**--spacing**：默认单位 **Å**（不再使用 nm）。  
  - `radius`：纤维半径，用于与体积卷积的管状核及 MOD 过滤时的距离阈值，默认 250 Å。  
  - `spacing`：等间距采样步长，默认 40 Å。
- **--rec**（**必选**）：输入的 .rec 体积路径。当前仅支持 .rec，不使用 convmap.mrc。
- **--curvature**：曲率 [0,1]，0=最直、1=最弯，默认 0.1。
- **--threshold**：卷积响应二值化百分位 (0–100)，默认 75。
- **--erode**：二值化后的形态学侵蚀迭代次数，纤维不连续时可试 1–3，默认 0。
- **--curve-points**：每条拟合曲线的采样点数，默认 50；减小可加快运行。
- **--fiber-index**：逗号分隔的 0-based 索引（如 `0,2,5` 或 `0,1-3`），仅拟合并输出这些纤维；不指定则拟合全部。
- **--workers**：并行拟合的进程数，默认 1；自动不超过待拟合纤维数；在 ChimeraX 中强制为 1。
- **--invert**：归一化后取反密度，使**暗纤维变亮**（cryo-ET 中纤维常为低密度），再做增强与卷积。
- **--enhance**：`conv`（默认）= 圆柱核卷积；`frangi` = Hessian 管状增强（基于二阶导，对 3D 较快，纤维处变亮）。
- **--kuwahara-radius**：3D Kuwahara 降噪半宽（像素）；**默认 0（关闭）**，因 3D 极慢；需边缘保留时可设为 3 等。

## 流程概要

1. 读取 STAR，**对每一对起止点**（连续两行）做纤维拟合；若最后只剩一行（仅起始点），**打印提醒**，其余成对照常处理。
2. 拟合方式（参考 FibrilFinder）：在起止点附近裁剪 ROI，用**用户指定半径**的纤维（管状核）与体积**卷积** → **二值化** → [可选侵蚀] → **骨架化**，以二值化体的**骨架中心点为中心线** → 用**可调节曲率**的样条拟合。
3. 每条拟合曲线保存为 **.axm**（ArtiaX CurvedLine 格式，npz），可在 ChimeraX+ArtiaX 中打开；若指定 **--fiber-index N**，则只拟合并保存第 N 条纤维的 .axm。
4. **指定 --out 时**，当前拟合的纤维（全部或 --fiber-index 指定的一条）按两种模式之一输出颗粒，二选一：
   - **等间距模式**（默认）：按 `--spacing`（Å）沿每条曲线等间距采样，角度从起点到终点线性插值，**所有纤维**的颗粒合并写入一个 STAR。
   - **MOD 过滤模式**：指定 `--mod`，对**每条**曲线保留到曲线距离 ≤ `--radius`（Å）的颗粒，合并写入一个 STAR。

## 1. 在 ChimeraX 中运行

需先安装 ArtiaX。在 ChimeraX 中：

```text
runscript path/to/filament/fiber_to_star.py --star path/to/data/cilia_106003_1_bin6.star --rec path/to/cilia_106003_1_bin6.rec --voxel-size 14.5
```

必选：`--rec`、`--voxel-size`。示例（指定第 1 条纤维、调节曲率并输出 STAR）：

```text
runscript .../fiber_to_star.py --star .../cilia_106003_1_bin6.star --rec .../cilia_106003_1_bin6.rec --voxel-size 14.5 --fiber-index 1 --curvature 0.2 --curve-points 50 --spacing 40 --out particles.star
runscript .../fiber_to_star.py --star .../cilia_106003_1_bin6.star --rec .../cilia_106003_1_bin6.rec --voxel-size 14.5 --fiber-index 1 --mod .../cilia_106003_1_bin6.mod --radius 250 --out filtered.star
```

## 2. 独立运行（不打开 ChimeraX）

```bash
pip install -r filament/requirements.txt
python filament/fiber_to_star.py --star filament/data/cilia_106003_1_bin6.star --rec filament/data/cilia_106003_1_bin6.rec --voxel-size 14.5 --spacing 40 --out particles.star --no-open
```

仅生成 .axm 与 STAR，不调用 ChimeraX。

## 输出

- **.axm**：每条纤维一个 `*_fiberN.axm`（ArtiaX CurvedLine 格式，内容为 npz），**文件扩展名为 .axm**。在 ChimeraX 中需通过 **ArtiaX 面板 → Geometric Models → Open Geomodel...** 选择该 .axm 文件打开曲线；若使用 `--fiber-index N` 则只生成该条的 .axm。
- **STAR**（仅当指定 `--out` 时输出，**包含当前拟合的全部纤维**的颗粒）：
  - **等间距模式**：`_rlnCoordinateX/Y/Z` 为脚本沿曲线等间距采样的坐标；`_rlnAngleRot/Tilt/Psi` 从该条纤维**起始点角度线性过渡到终止点角度**；`_rlnOriginX/Y/Z` 为 0。
  - **MOD 过滤模式**：`_rlnCoordinateX/Y/Z` 为 MOD 中保留颗粒的**原始坐标**；MOD 无角度与 origin 信息，故 `_rlnOriginX/Y/Z` 为 0，`_rlnAngleRot/Tilt/Psi` 填该条纤维的起始点角度（同一条纤维内一致）。

## 注意

- 当前实现**仅考虑 .rec** 作为体积输入（convmap.mrc 与 rec 相关系数较低且非所有用户都有 convmap）。
- 纤维信号若不连续，可尝试增大 `--radius` 或使用 `--erode`（二值化后侵蚀迭代次数，如 1–3）。
