<a id="surface-postprocessing"></a>

# 皮层表面后处理

**从四份皮层表面出发，完成即插即用的形态测量与皮层分区。**

[DDSurfer](../README.zh-CN.md) | [处理流程](#workflow) | [运行](#run) | [表面要求](#surface-requirements) | [输出](#outputs) | [优化措施](#optimizations-over-freesurfer) | [English](README.md)

可使用 DDSurfer 或其他方法重建的皮层表面，无需 DWI、DTI 或结构 MRI。
主流程默认关闭后处理；添加 `--post-process` 后，会在表面转换回原生空间后启用此阶段。

---

<a id="workflow"></a>

## 处理流程

后处理的起点是**原生空间的白质与软脑膜表面**：可以是 DDSurfer 完成 MNI 到原生空间
转换后的结果，也可以是其他方法得到的等效表面。此阶段不重新执行皮层表面重建，
也不将脑解剖结构配准到 MNI。下文的球面配准用于在表面之间传递图谱信息。

```text
原生空间的白质／软脑膜表面
  -> 准备参考体积，将 scanner RAS 转为 surface RAS
  -> 计算表面面积与曲率
  -> 表面膨胀与沟深计算
  -> 生成球面并进行球面图谱配准
  -> 计算厚度，生成皮层标签与皮层分区
  -> 导出脑区标签、统计表及 fsaverage 表面标量数据
```

| 阶段 | 处理内容 | 主要结果 |
| --- | --- | --- |
| 准备 | Python 检查白质与软脑膜表面的顶点对应关系，准备 LIA 方向的参考体积并转换坐标。 | `mri/brain.mgz`、`mri/orig.mgz`、FreeSurfer 白质／软脑膜表面及空间质控记录。 |
| 测量 | FreeSurfer 计算白质与软脑膜表面的面积、平均曲率、高斯曲率及平滑曲率。 | `area`、`area.pial`、`white.H/K`、`pial.H/K`、`curv`、`curv.pial`。 |
| 膨胀 | 运行 `mris_inflate`，随后测量膨胀表面的曲率。 | `inflated`、`sulc`、`inflated.H/K`。 |
| 配准 | 使用 `mris_sphere` 和 `mris_register` 建立与图谱的球面对应关系。 | `sphere`、`sphere.reg`。 |
| 传递 | FreeSurfer 测量厚度，并传递 fsaverage 的皮层标签及所选分区标注。 | `thickness`、`cortex.label`、`aparc.annot`、`aparc.a2009s.annot`。 |
| 汇总 | Python 导出脑区标签与统计表；一条 FreeSurfer 命令映射四份标量数据。 | 脑区 `.label`、`.stats` 文件，以及 fsaverage `.mgh` 数据。 |

左右半球均遵循上述顺序，默认在总线程预算内并行执行；`--serial` 改为顺序执行。
原生白质与软脑膜表面的解剖几何保持不变；膨胀表面和球面只是额外的表示形式，
不会替换用户提供的表面。

当安装中提供 `mris_place_surface` 时，测量命令参考
[FastSurfer recon-surf](https://github.com/Deep-MI/FastSurfer/blob/dev/recon_surf/recon-surf.sh)；
旧版 FreeSurfer 则使用兼容路径。所选图谱的标签通过球面配准传递，
不是根据组织分割推断得到。

---

<a id="requirements"></a>

## 环境与输入要求

| 组件 | 配置要求 |
| --- | --- |
| FreeSurfer | 配置 `FREESURFER_HOME`、`FS_LICENSE` 及安装中的 `fsaverage`。 |
| Python 包 | 使用根目录统一的 [`environment.yml`](../environment.yml) 或 [`requirements.txt`](../requirements.txt)。 |
| 皮层表面 | 左右半球的白质与软脑膜表面；同一半球的两份表面须具有一致的拓扑。 |

详见根目录的 [依赖配置](../README.zh-CN.md#key-dependencies)。

启用后处理时，主流程会在预处理前检查所需的 FreeSurfer 可执行文件、许可证和图谱文件。
未启用 `--post-process` 的表面推理不需要这些后处理检查。
复用最终结果时，也会验证后处理输出的完整性。

---

<a id="run"></a>

## 运行

<a id="with-ddsurfer"></a>

### 随 DDSurfer 主流程运行

```bash
python3 run_ddsurfer_pipeline.py --subject <subID> --post-process
```

<a id="independently"></a>

### 独立运行

```bash
bash postprocessing/run.sh --subject <subID> \
  --lh-white ./outputs/<subID>/ddsurfer/lh.white.obj \
  --lh-pial ./outputs/<subID>/ddsurfer/lh.pial.obj \
  --rh-white ./outputs/<subID>/ddsurfer/rh.white.obj \
  --rh-pial ./outputs/<subID>/ddsurfer/rh.pial.obj \
  --output-root ./outputs
```

| 独立运行选项 | 行为 |
| --- | --- |
| `--hemi left`、`right` 或 `both` | 选择半球，默认双侧。 |
| `--atlases aparc` | 选择图谱，默认 `aparc,aparc.a2009s`。 |
| `--brain-source <file>` | 可选的原生空间标量 MRI，用于准备 FreeSurfer 参考体积。 |
| `--threads 4` | 总线程预算，在左右半球之间分配。 |
| `--serial` | 顺序处理两个半球，不并行执行。 |
| `--freesurfer-home <path>` | 覆盖 `FREESURFER_HOME`。 |

主流程对应选项为 `--postprocess-hemi`、`--postprocess-atlases`、
`--postprocess-threads` 和 `--postprocess-serial`。
主流程默认使用 DTI 估计阶段得到的原生空间 b0 图像作为参考。

---

<a id="surface-requirements"></a>

## 表面要求

输入须为**原生 scanner RAS 坐标系中的封闭三角网格，坐标单位为毫米**。
同一半球的白质与软脑膜表面必须具有一致的顶点索引和三角面片。

| 格式 | 注意事项 |
| --- | --- |
| OBJ / PLY / OFF | 保存顶点索引的表面格式；需保持白质与软脑膜表面的对应关系。 |
| FreeSurfer 二进制表面 | 必须包含有效的体积几何信息。 |
| STL | 配对文件须保留相互对应的三角面片顺序；程序使用共享的顶点合并映射重建索引。 |

对应关系不合法时会直接拒绝，不通过最近邻匹配猜测对应关系。
DDSurfer 默认输出 OBJ 格式。

<a id="coordinate-handling"></a>

### 坐标处理

没有 MRI 时，程序根据表面构建仅含几何信息的 LIA 参考体积。
其中没有解剖图像强度，质控日志会标记为 `geometry_only`。
提供 `--brain-source` 时，将原生图像轴向重定向为 LIA：
轴对齐输入使用轴交换或翻转，斜轴输入使用重切片。
这两条路径都不会将图像配准到模板。

程序通过 MRI 的仿射矩阵和 `vox2ras_tkr`，将 scanner RAS 转换为 FreeSurfer surface RAS。
改变的是坐标表示，而不是解剖位置。LIA 参考体积保留球面配准所需的解剖轴向约定；
仅检查坐标往返变换，不能验证该轴向约定是否正确。
详见 [FreeSurfer 坐标系统](https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems)。

---

<a id="outputs"></a>

## 输出结果

结果直接保存到 `outputs/<subID>/`：

| 目录 | 代表文件 | 用途 |
| --- | --- | --- |
| `surf/` | `lh.white`、`lh.pial`、`lh.curv`、`lh.sulc`、`lh.thickness`、`lh.area` | 原生空间表面与形态测量。 |
| `surf/` | `lh.inflated`、`lh.sphere`、`lh.sphere.reg` | 表面膨胀与球面图谱配准。 |
| `label/` | `lh.cortex.label`、`lh.aparc.annot`、`lh.aparc.a2009s.annot` | 皮层掩膜、分区及脑区标签。 |
| `stats/` | `lh.aparc.stats`、`lh.aparc.a2009s.stats` | 脑区表面统计。 |
| `fsaverage/` | `lh.thickness.mgh`、`lh.curv.mgh`、`lh.sulc.mgh`、`lh.area.pial.mgh` | 映射到标准表面的标量数据。 |
| `mri/` | `brain.mgz`、`orig.mgz` | FreeSurfer 参考体积。 |
| `logs/` | 命令日志、耗时记录与坐标检查。 | 执行记录与质控。 |

右半球文件使用 `rh.` 前缀。实际输出取决于所选图谱和半球。
输出根目录同时作为 FreeSurfer 的 `SUBJECTS_DIR`，并包含指向安装中 `fsaverage` 模板的链接；
该链接与个体目录下保存标量数据的 `fsaverage/` 不是同一个目录。

<a id="interpreting-the-measurements"></a>

### 测量结果的含义

- 曲率、面积和厚度均测量用户提供的表面；测量命令不会移动这些表面。
- 厚度计算使用 FreeSurfer 默认的 5 mm 上限。
- 脑区统计表在传递得到的皮层标签范围内，报告顶点数、白质／软脑膜表面面积、
  厚度均值与标准差，以及曲率绝对值的均值。
- 这些是**仅基于表面的统计量**，不是完整的 `recon-all` 统计表；
  不推断 `GrayVol`、eTIV 或依赖组织分割的体积指标。
- 仅含几何信息的 `mri/` 参考体积不是重建出来的结构图像。

---

<a id="optimizations-over-freesurfer"></a>

## 相较 FreeSurfer 的流程优化

DDSurfer 保留 FreeSurfer 的核心皮层表面算法，同时减少外围的重复工作。
部分改进替换单条命令，部分改进任务调度和结果复用。
这是一套**仅基于表面的分析流程**，并不等于加速后仍提供完整 `recon-all` 的所有输出。

| 优化措施 | 实现方式 | 收益 |
| --- | --- | --- |
| Python 脑区标签导出 | 根据分区标注选择顶点，按相同标签文本格式写出，替代 `mri_annotation2label`。 | 省去一个 FreeSurfer 子进程；本地测试中，35 个标签的导出耗时由 **1.13 s 降至 0.37 s**。 |
| 共享 fsaverage 映射 | 将厚度、曲率、沟深和软脑膜表面面积合并为一次 `mri_surf2surf --split` 调用，而不是分别调用四次。 | 共享球面读取及对应关系计算；受测阶段由 **13.68 s 降至 3.23 s**，耗时约减少 **76%**。完整 MGH 元数据仍由 FreeSurfer 写出。 |
| 进程内统计 | 直接调用已有的 Python 表面统计表写入函数。 | 避免启动另一个 Python 解释器及重复导入依赖；测量定义不变。 |
| 同一半球共享输入 | 按需读取并复用白质表面顶点、皮层顶点索引及测量数组，用于多个图谱、标量打包和厚度检查。 | 默认双图谱汇总阶段，Python 标量图读取次数由 **15 次降至 6 次**；白质网格和皮层标签读取分别由 **2 次降至 1 次**。 |
| 有界标签文本写入 | 每块最多写入 4,096 行，不构建完整的五列文本导出中间数组。 | 保持输出格式一致；与前一版 Python 写入实现相比，35 个标签的耗时由 **0.35 s 降至 0.26 s**，两者均包含几何读取。 |
| Python 几何准备 | 在进程内完成格式、坐标转换及参考体积准备，不再调用独立转换命令。 | 减少转换子进程，同时增加显式的拓扑、视野范围和坐标往返检查。 |
| 受控双半球并行 | 在左右半球之间分配总线程预算。 | 重叠执行相互独立的工作，而不将配置的线程总数翻倍。 |
| 校验和续跑与运行锁 | 复用验证通过的阶段，并在写入正式位置前验证 Python 临时输出。 | 减少重复计算，避免重复处理同一个体；残缺或损坏结果会重试，不会误判为完成。 |
| 按需执行旧版测量兼容路径 | 仅在旧版安装的面积结果需要重建时加载白质／软脑膜网格，并验证曲率复制结果后再复用。 | 已完成的兼容阶段续跑时，不再重新读取两份网格或改写未变化的曲率文件。 |

直接使用已重建的表面，也省去了上游体素分割与表面重建，减少了此任务的处理工作量。
但因此不会产生依赖分割的 `GrayVol`、eTIV 等指标。
NumPy 顶点面积兼容实现仅用于旧版安装；新版安装仍使用 FreeSurfer 的面积计算。

**保持不变的部分：** 曲率、厚度、表面膨胀、球面生成与配准、皮层标签传递及图谱映射，
均保留原有 FreeSurfer 算法和参数；不会为了提速减少迭代次数、修改平滑设置或降低厚度上限。

最近的共享读取与分块写入改进，使受测双图谱 Python 汇总阶段的耗时由
**0.88 s 降至 0.65 s**，约减少 **26%**；结果为三次文件缓存预热后测试的中位数。
该阶段包括标签导出、脑区统计、标量打包和厚度检查，不包括 FreeSurfer 配准或映射。
读取次数指 Python 数据读取函数的调用次数，不包括校验和计算等全部文件系统操作。

<a id="verified-improvements-and-limits"></a>

### 验证结果与适用范围

- **受测输出等价：** 219 份脑区标签文件和全部四份 fsaverage MGH 数据，
  与分别执行对应 FreeSurfer 命令的结果逐字节一致。
  脑区统计与已有的 Python 表面统计写入实现逐字节一致。
- **共享输入验证：** 另一次双图谱处理产生的 113 份标签、统计及打包数据文件，
  与前一版实现逐字节一致；四份映射后的 MGH 文件也再次与独立 FreeSurfer 调用核对。
- **实测范围：** FreeSurfer 7.4.1、个体 `100307` 的左半球结果及 fsaverage 标签案例，
  FreeSurfer 线程预算为 2。数据是阶段级耗时，不构成整个流程的加速保证。
- **剩余开销：** 球面生成与配准未改变，仍是主要瓶颈。尚未测量峰值内存是否降低；
  批量映射会同时保存四个标量场，共享读取也会在每个半球处理中临时保留少量 CPU 数组，
  并非持续预加载多个个体。
- **回归覆盖：** 153 项 CPU 测试通过，覆盖调度、共享读取、新旧测量路径的续跑，
  以及无效或缺失输出的处理。改动阶段已使用真实 FreeSurfer 命令验证；
  端到端流程编排使用模拟命令进行测试。

实现细节、源码参考及回归测试命令见 [优化与等价性验证](OPTIMIZATIONS.zh-CN.md)。

---

<a id="reuse-and-execution"></a>

## 复用与执行

完成阶段经校验和验证后可以续跑。输入或设置变化时，需使用新的输出目录。
检查输出时也会核对上游依赖；命令失败会终止流程，不会记录为成功。

两个半球共享总线程预算，各自保存命令日志。不希望并行执行时，可使用 `--serial`。
