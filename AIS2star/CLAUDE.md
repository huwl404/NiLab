# Project: AIS2star

cryo-ET 颗粒提取。输入 AIS 概率图 `.mrc`，输出 RELION `_particles.star`。

- `mem2star.py` — prob-ridge 直接提 midband → junction_cut。
- `fiber2star.py` — K-mode 方向检测 → skel + junction split → per-mode greedy spline 提纤维。

GPU 驻留（cupy / cucim），仅 I/O 边界才 H2D/D2H。默认 bin=2；单膜厚 ~100 Å（~5 vox after bin）。

**硬件**：单卡 8 GB VRAM。每加一个与 vol 同 shape 的 label/mask 体积都要核算峰值（300×1024² float32 = 1.26 GB）。int8 / uint16 / uint8 优先；`save_mrc` 前不要在 GPU 侧 `.astype(cp.float32)`（内部会 CPU 侧转 dtype）。

## 代码风格

**函数调用**
- ≤3 个参数单行写；多参数才分行
- `save_mrc(path, data, voxel)` 等debug调用函数一律单行，不分行

**Debug MRC 保存**
- 各阶段 debug 输出由**产出它的函数**通过 `debug_prefix` 参数保存。文件名 `_NN_stage.mrc` 顺序编号

**CLI**
- 一个 `p.add_argument` 一行，不折行，使用简洁英文
- `help=` 只说**调这个会怎样**（"大 = 更激进"），不解释机理；机理写进函数 docstring
- 不把内部常量藏成模块级 `_CONST = ...`：要么 CLI 参数（带"一般不调"），要么 inline
- 不加不影响输出的参数

**docstring / 注释**
- 讲**为什么 / 原理**，不重复代码做了什么，使用中文
- 不写 PR-style 注释（"用于 X 流程"、"修复 issue #N"），它们会腐烂

**禁止**
- 为不可能情况做 error handling / fallback

**GPU 库优先**
- 遇到 numpy / scipy / skimage 调用时先查是否有 cupy / cupyx / cucim 对应（如 `scipy.spatial.cKDTree` → `cupyx.scipy.spatial.KDTree`；`scipy.ndimage.{sum,maximum,minimum}_labels` → `cupyx.scipy.ndimage.*`）

## 协作习惯

- **非 trivial 任务：先讨论，不要直接动手**。列方案 + 权衡 + 推荐选项，等我拍板
- 我给一组反馈时：**你先分析优先级再按序推进**
- 不拘技术债：架构性重写 OK，不用为了"已验证"死守旧架构，但必须等我拍板
- 末尾总结**改了什么 + 下一步测什么**
- 版本迭代：我会写到 `xxx_v2.py`，验收后自己改名覆盖主文件；你不用主动归档旧版

## 尺度参考（默认值所在）

| 物理量 | 值 |
|---|---|
| voxel（bin 后） | ~20 Å |
| 单膜厚度 | 100 Å ≈ 5 vox |

