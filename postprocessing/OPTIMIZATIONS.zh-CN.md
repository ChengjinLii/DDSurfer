<a id="postprocessing-optimizations"></a>

# 皮层表面后处理优化

**减少重复工作，保留表面几何与测量定义。**

[使用指南](README.zh-CN.md) | [Python 替代实现](#python-replacements) | [保留的算法](#retained-algorithms) | [验证](#validation) | [English](OPTIMIZATIONS.md)

---

<a id="python-replacements"></a>

## Python 替代实现

| 操作 | 实现 | 范围 |
| --- | --- | --- |
| 分区标注转脑区标签 | [`labels.py`](labels.py) 替代 `mri_annotation2label`。 | 使用分区颜色编码和白质表面坐标；保留顶点顺序、surface RAS 坐标、文本精度，以及标注中实际存在的 `unknown` 标签，不改变分区。 |
| 脑区统计 | 在当前进程内调用 [`stats.py`](stats.py)，不另启 Python 解释器。 | 使用相同的表面统计表写入实现和测量定义，不新增指标。 |
| 共享测量输入 | [`pipeline.py`](pipeline.py) 仅在重建统计或标签时建立当前半球的输入缓存。 | 不同图谱复用白质顶点和皮层／测量数组；标量打包及厚度检查共享相关测量值，保留原数据类型和聚合计算操作。 |
| 标签文本写入 | [`labels.py`](labels.py) 每个文本块最多写入 4,096 行。 | 避免构建完整的五列中间数组及通用 `numpy.savetxt` 的逐行循环，同时保留文本精度与行顺序。 |
| 表面／参考体积转换 | 已有的 [`geometry.py`](geometry.py) 使用 nibabel、NumPy 和 trimesh。 | 读取支持的网格、准备 FreeSurfer 参考体积、将 scanner RAS 转为 surface RAS，并检查拓扑及坐标往返，不需要转换子进程。 |
| 旧版安装的面积计算 | 已有的 NumPy 顶点面积兼容实现。 | 仅在 `mris_place_surface` 不可用时使用；新版 FreeSurfer 的面积计算路径保持不变。 |

脑区统计表**不能替代完整的 `mris_anatomical_stats` 或 `recon-all` 结果**。
它们在皮层标签范围内报告顶点数量、白质／软脑膜表面面积、厚度均值与标准差，
以及曲率绝对值的均值；不估计 `GrayVol`、eTIV 或依赖分割的组织体积。

---

<a id="shared-mapping-and-reliable-reuse"></a>

## 共享映射与可靠复用

- **一次映射替代四次。** [`overlays.py`](overlays.py) 将厚度、曲率、沟深和软脑膜表面面积
  堆叠为四帧，由一次 `mri_surf2surf` 调用共同映射；配准和默认映射设置与之前相同。
- **输出仍由 FreeSurfer 写入。** `--split` 分别写出标量 MGH 文件。
  Python 验证全部四份输出后，将其复制到临时输出位置，不丢弃 FreeSurfer 的额外 MGH 元数据。
- **受控双半球并行。** 已有调度器在两个半球间分配总线程预算，而不是为每个半球都分配全部预算。
- **按内容校验续跑。** 是否复用由输入、实现代码和输出的校验和决定。
  标签缺失或损坏时，在 Python 中重建；标量映射结果缺失或损坏时，重新执行一次共享映射。
  Python 阶段验证临时输出后才写入正式位置并记录成功；写入中断时没有有效完成记录，下次会重试。
- **按需执行旧版兼容路径。** 仅在面积重建回调中加载原始白质／软脑膜网格对，
  不在检查已完成阶段前预先加载。原有的平均曲率兼容复制改为经校验和验证的 Python 阶段，
  不再无条件改写文件；面积公式和曲率值均不变。

共享输入数组只存在于当前半球的本次处理过程，不是持久化的个体缓存。
它以少量临时 CPU 内存换取更少的文件读取与解码，不据此声称峰值内存降低。
续跑时，已完成的统计阶段不会触发提前加载测量数组。

个体运行锁和已有空间检查均保留。输入表面只读取、不修改。
这些优化不新增依赖，也不要求用户调整命令行选项。

---

<a id="retained-algorithms"></a>

## 保留的算法

| 操作 | FreeSurfer 命令 |
| --- | --- |
| 白质／软脑膜表面面积与平滑曲率 | `mris_place_surface --area-map` / `--curv-map` |
| 平均曲率与高斯曲率 | `mris_curvature` |
| 表面膨胀与沟深 | `mris_inflate` |
| 球面生成与配准 | `mris_sphere`、`mris_register` |
| 厚度 | `mris_place_surface --thickness`；兼容路径使用 `mris_thickness` |
| 皮层标签传递 | `mri_label2label` |
| 图谱与标量数据传递 | `mri_surf2surf` |

皮层标签和图谱传递不使用 Python 最近邻替代实现。
配准迭代次数、平滑设置及厚度上限均未降低。
NumPy 面积实现已与新版 FreeSurfer 输出比较：受测案例中，
单顶点差异最大约为 `4.8e-7 mm^2`。尽管很小，仍保留新版面积命令，
不为省去一个耗时较短的子进程而引入数值变化。

未替换球面／配准算法，也未采用自定义皮层标签传递方案，
因为尚未证明这些替代方案能保持输出等价，或提高图谱准确性。
本轮额外优化改变的是数据处理方式，而非这些算法。

---

<a id="validation"></a>

## 验证

真实命令验证使用 **FreeSurfer 7.4.1**、个体 `100307` 的左半球结果，
以及安装中的 fsaverage 标签。原有结果没有被修改。

| 验证项 | 结果 |
| --- | --- |
| 脑区标签导出 | `aparc` 和 `aparc.a2009s` 共 219 份文件，包含左右 fsaverage 案例；文件集合一致，各标签文件逐字节一致。 |
| 四份 fsaverage 标量数据 | 与四次独立映射命令产生的 MGH 文件逐字节一致；数值、形状、仿射矩阵、文件头及附加元数据均保留。 |
| 脑区统计 | 与已有的 Python 表面统计表写入结果逐字节一致。 |
| 共享输入汇总 | 两个图谱的 113 份标签、统计及打包数据文件与前一版实现逐字节一致；四份映射后的 MGH 文件再次与独立 FreeSurfer 调用核对。 |
| CPU 回归测试 | 153 项测试通过，覆盖环境检查、双半球调度、共享输入读取、新旧测量路径续跑、异常输入及缺失／损坏输出处理。 |

FreeSurfer 线程预算为 2 时，代表性阶段的耗时如下：

| 阶段 | 优化前 | 优化后 |
| --- | --- | --- |
| 导出 35 个 `aparc` 标签 | 1.13 s | 0.37 s |
| 映射四份 fsaverage 标量数据 | 13.68 s | 3.23 s |

另一次对比以前一版 Python 实现为基准，在同一半球上进行三次文件缓存预热后的测试。
两版保留相同的数值运算与输出格式：

| 操作 | 优化前 | 优化后 |
| --- | --- | --- |
| 导出 35 个标签，含白质网格读取 | 0.35 s | 0.26 s |
| 双图谱标签／统计导出、标量打包与厚度检查 | 0.88 s | 0.65 s |
| 上述汇总阶段的 Python 标量图读取次数 | 15 | 6 |
| 上述汇总阶段的白质网格／皮层标签读取次数 | 2 / 2 | 1 / 1 |

这些次数统计的是数据读取函数调用，而非全部文件系统读取；校验和验证仍会执行。
测试还确认：旧版测量输出全部可复用时，不会重新加载白质／软脑膜网格，
也不会改写未变化的曲率文件。

这些是本地阶段级测量，**不构成整个流程的加速保证**。
球面生成与配准仍是主要耗时部分，且未作改动。
修改阶段使用真实 FreeSurfer 命令验证；完整后处理流程使用模拟命令进行回归测试，
本轮耗时比较没有从头到尾重跑全部阶段。其他 FreeSurfer 版本尚未进行性能测试。

在仓库根目录运行 CPU 检查：

```bash
CUDA_VISIBLE_DEVICES='' python3 -B -m unittest discover -s tests -q
```

<a id="references"></a>

### 参考源码

实现已对照安装版本所对应的 FreeSurfer 源码修订进行核查：
[分区标注转标签](https://github.com/freesurfer/freesurfer/blob/7eb8460/mri_annotation2label/mri_annotation2label.cpp)、
[标签格式写入](https://github.com/freesurfer/freesurfer/blob/7eb8460/utils/label.cpp)，
以及 [多帧表面映射](https://github.com/freesurfer/freesurfer/blob/7eb8460/mri_surf2surf/mri_surf2surf.cpp)。
