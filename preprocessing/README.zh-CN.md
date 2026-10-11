<a id="preprocessing"></a>

# 预处理

**从已校正的扩散磁共振数据，生成表面推理所需的输入。**

[DDSurfer](../README.zh-CN.md) | [运行](#run) | [脑掩膜生成](#optional-brain-masking) | [输出](#outputs) | [English](README.md)

DDSurfer 主流程会自动执行此阶段。`run.sh` 先通过 `dti.sh` 估计 DTI，
再准备配准后的推理输入体积。

---

<a id="requirements"></a>

## 输入与环境要求

| 输入 | 要求 |
| --- | --- |
| DWI | 已校正的四维 NIfTI，包含 b0 和扩散加权图像。 |
| bval / bvec | 梯度表与 DWI 体积数量一致；bvec 可为 3×N 或 N×3。 |
| 脑掩膜 | 非空的三维 NIfTI，与 DWI 的原生体素网格和物理空间一致。 |

头动、涡流和磁敏感畸变校正，以及相应的梯度更新，须提前完成。
标量参数图或已拟合的张量不能作为原始 DWI 输入。源文件不会被修改。

Python 依赖通过根目录的 [`environment.yml`](../environment.yml)
或 [`requirements.txt`](../requirements.txt) 安装。
还需单独安装 Slicer 及 SlicerDMRI，详见 [依赖配置](../README.zh-CN.md#key-dependencies)。

主入口会在预处理前检查输入、模型、设备支持和 Slicer 命令行模块。
中间缓存被清理后，仍可通过主流程的 `--resume` 复用经过验证的最终结果。

---

<a id="run"></a>

## 运行

<a id="complete-volume-preprocessing"></a>

### 完整体积预处理

```bash
bash preprocessing/run.sh --subject <subID> --raw-input-root ./inputs
```

<a id="dti-estimation-and-registration-only"></a>

### 仅执行 DTI 估计与配准

```bash
bash preprocessing/dti.sh --subject <subID> \
  --dwi ./inputs/<subID>/dwi.nii.gz \
  --bval ./inputs/<subID>/dwi.bval \
  --bvec ./inputs/<subID>/dwi.bvec \
  --mask ./inputs/<subID>/mask.nii.gz \
  --output-root ./outputs/<subID>/.cache/dti
```

自动查找输入时，支持默认的 `inputs/<subID>/` 文件组合、以个体编号命名的文件，
以及 HCP 的 `T1w/Diffusion/` 目录结构。其他结构请显式指定文件路径。
两个入口均可通过 `--help` 查看选项及独立运行时的输出路径。

<a id="workflow"></a>

### 处理流程

| 步骤 | 操作 |
| --- | --- |
| 1 | 验证图像几何信息、梯度数量，以及张量拟合设计矩阵的秩。 |
| 2 | 将 DWI 转为 NHDR，使用加权最小二乘法（WLS）拟合张量，并提取 b0。 |
| 3 | 在原生空间计算 FA、MD、特征值和 Trace。 |
| 4 | 将 b0 配准到图谱，并对各标量参数图应用同一变换。 |
| 5 | 对推理输入施加掩膜、重采样，并进行 Z-score 标准化。 |

`--minimal` 仅准备图谱空间中的五个推理通道，跳过模型不使用的标准化 DTI 缓存图。
原生空间的 Trace 仍会计算。主流程使用此模式；独立脚本默认保留完整的标量输出。

---

<a id="optional-brain-masking"></a>

## 可选脑掩膜生成

自动生成脑掩膜**默认关闭**。在主流程中启用 `--auto-mask` 后，可不提供掩膜文件：

```bash
python3 run_ddsurfer_pipeline.py --subject <subID> --auto-mask
```

此方法先平均 DWI 中 `b <= 50` 的体积，再对得到的原生空间 b0 图像运行
FreeSurfer `mri_synthstrip`。
该做法遵循 [MRtrix3 脑掩膜指南](https://userdocs.mrtrix.org/en/dev/dwi_preprocessing/masking.html)
介绍的平均 b0 掩膜生成思路；本实现不依赖 MRtrix3。

**配置与行为**

- 安装包含 `mri_synthstrip` 及其模型权重的 FreeSurfer。
- 设置 `FREESURFER_HOME`，或传入 `--freesurfer-home <path>`。
- 使用 CPU 生成掩膜；`--mask-threads` 默认值为 4。
- 用户提供的掩膜优先，不会被替换。
- 生成的二值掩膜必须与 DWI 的形状和物理空间仿射矩阵匹配。
- 应检查掩膜的解剖质量；仅通过几何检查并不能保证掩膜正确。

生成的掩膜保存为 `outputs/<subID>/dti/<subID>-brainmask.nii.gz`。
`logs/mask.json` 记录输入、生成方法和校验和；输入未变化时，清理中间缓存后仍可复用掩膜。
如果输入变化，或生成的掩膜被手动编辑，需使用新的输出目录。

也可以独立生成掩膜，再将其传给 `run.sh` 或 `dti.sh`：

```bash
python3 preprocessing/brain_mask.py \
  --dwi ./inputs/<subID>/dwi.nii.gz \
  --bval ./inputs/<subID>/dwi.bval \
  --bvec ./inputs/<subID>/dwi.bvec \
  --output ./outputs/<subID>/dti/<subID>-brainmask.nii.gz \
  --work-dir ./outputs/<subID>/.cache/mask \
  --record ./outputs/<subID>/logs/mask.json
```

此功能不需要 T1 图像，也不需要运行 `recon-all`。
自动生成的掩膜可能与数据集提供的掩膜不同，从而影响后续结果。

---

<a id="outputs"></a>

## 输出结果

| 主流程中的位置 | 内容 | 默认保留 |
| --- | --- | --- |
| `outputs/<subID>/dti/` | 原生空间的 FA、MD、三张特征值图和 Trace，以及可选的自动生成掩膜。 | 是 |
| `outputs/<subID>/logs/` | 输入记录、原生 DTI 摘要和配准变换。 | 是 |
| `outputs/<subID>/.cache/dti/` | 张量、b0 和配准后的标量参数图。 | 否 |
| `outputs/<subID>/.cache/volumes/` | 已施加掩膜并标准化的推理输入体积。 | 否 |

`export.py` 将原生标量参数图导出为 NIfTI，并核对其几何信息与原始 DWI 是否一致。
导出过程不进行重采样、掩膜处理或标准化。没有 Trace 的旧缓存仍可导出已有的五张参数图。

图谱到原生空间的变换保存在 `logs/b0ToAtlasT2.tfm`。
主流程成功后会删除中间文件，除非指定 `--keep-cache`；失败时保留缓存以便排查。

---

<a id="performance-and-reuse"></a>

## 性能与复用

- 独立脚本通过 `--jobs 2` 并行执行相互独立的 DTI 阶段，主流程对应选项为
  `--preprocess-jobs 2`。默认并发数为 1；应根据可用 CPU 和内存控制并发。
- 张量拟合与配准仍按依赖顺序执行。插值方式、配准设置及两次重采样步骤均保持不变。
- 已完成阶段仅在输入、输出校验和验证通过后复用；原子写入避免将未写完整的文件误判为有效结果。
- 标准化结果与未标准化中间文件分别保存，避免续跑时重复标准化。
- 原始输入不同，或旧结果没有运行记录时，应使用新的输出目录，不会静默复用无关结果。

<a id="helpers"></a>

### 辅助脚本

| 文件 | 用途 |
| --- | --- |
| `inputs.py`、`brain_mask.py` | 输入验证与可选脑掩膜生成。 |
| `volumes.py`、`mask.py`、`resample.py` | 掩膜处理与推理网格重采样。 |
| `normalize.py`、`normalize_dti.py` | 最终 Z-score 标准化，以及独立的 DTI 缓存标准化。 |
| `export.py`、`cache.py` | 原生标量导出与经验证的阶段复用。 |

两份标准化脚本用于不同阶段，其掩膜和背景处理方式不能互换。
