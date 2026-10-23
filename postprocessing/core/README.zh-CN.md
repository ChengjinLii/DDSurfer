# 本机编译皮层处理组件

**在用户本机编译原始皮层算法，也可选择已有的 FreeSurfer 安装。**

[后处理说明](../README.zh-CN.md) | [验证记录](VALIDATION.zh-CN.md) | [English](README.md)

不再随项目分发预编译的 Linux 可执行程序。Python 驱动七个原始 C++ 命令目标；
几何准备、脑区标签导出、表面统计、任务调度与续跑仍由 Python 完成。
图谱作为[独立数据资源](../assets/README.zh-CN.md)提供。

---

## 选择后端

| 后端 | 行为 |
| --- | --- |
| `auto` | 优先使用已配置的 FreeSurfer，否则使用已经完成的本机编译组件。 |
| `native` | 使用本机编译组件；后处理不采用 `FREESURFER_HOME` 指定的完整安装。 |
| `freesurfer` | 使用通过 `--freesurfer-home` 或 `FREESURFER_HOME` 指定的已有安装。 |

独立入口的 `--runtime` 可指定自定义本机组件目录；主入口对应为 `--postprocess-runtime`
和 `--postprocess-backend`。后处理仍默认关闭。普通推理和环境检查**不会自行下载或编译源码**；
未准备好时，会在数据处理前提示配置指令。

## 首次编译

准备 C/C++ 编译器、Git 和 CMake >=3.24,<4。Ubuntu／Debian 示例：

```bash
sudo apt-get install build-essential git libcrypt-dev
python -m pip install 'cmake>=3.24,<4'
python -m postprocessing.core.build --jobs 4
```

只有用户显式运行编译命令时，才会下载固定版本的 FreeSurfer、ITK 和 zlib 源码，
核对源码版本及压缩包 SHA256，编译所需原生命令并验证安装文件。
不会安装整个 FreeSurfer、GUI、其 Python 发行版或完整重建流程。
ITK 只启用本构建需要的 Common／VNL 基础部分，不启用全部模块。
对旧 ITK 编译器版本检查加入了兼容新 GCC 的修正；未重写数值计算函数或 FreeSurfer 算法源码。

编译仅使用 CPU；`--jobs` 控制并发，默认最多四个编译任务。首次构建需要网络及存放源码、
中间对象和依赖的磁盘空间，可能耗时数分钟或更久。编译脚本不会修改正在使用的 conda 环境，
也不会自行安装 Python 包。

默认在 `${XDG_CACHE_HOME:-~/.cache}/ddsurfer/postprocessing/` 下保存构建缓存和已安装组件。
可通过 `--work-dir`、`--prefix` 分别指定构建和安装目录，两者须分开；`--cmake` 可指定工具路径。
`--source-root` 接受固定版本的干净源码仓库；`--offline` 则要求该仓库及已校验的依赖压缩包
均已缓存。失败时保留日志与中间文件，便于排查和续跑。
七个命令全部编译并通过启动及完整性检查后，才发布可用组件；不覆盖已有安装。

目前以 Linux 为验证平台；**macOS 尚未验证，原生 Windows 不受支持**，因为上游代码使用
POSIX 接口。WSL2 属于另外的 Linux 运行方式，不等于原生 Windows 兼容。
提供源码并不意味着已经验证跨平台编译成功或数值一致。

## 使用

将自己的有效 FreeSurfer 激活文件放在源码及组件目录之外：

```bash
export FS_LICENSE=./private/freesurfer-license.txt

python run_ddsurfer_pipeline.py --subject <subID> --post-process \
  --postprocess-backend native
```

也可以独立处理已经生成的四份原生 scanner RAS 皮层表面：

```bash
bash postprocessing/run.sh --subject <subID> --native-root ./outputs \
  --output-root ./postprocessed --backend native --threads 4
```

若本机已经安装 FreeSurfer，直接沿用：

```bash
python run_ddsurfer_pipeline.py --subject <subID> --post-process \
  --postprocess-backend freesurfer --freesurfer-home ./software/freesurfer
```

已有安装在未设置 `FS_LICENSE` 时，也可读取其自身的 `license.txt`。
可选 SynthStrip 脑掩膜生成仍需独立配置；这些本机编译组件不包含 SynthStrip 及其模型。

## 保留的原生命令

| 命令 | 作用 |
| --- | --- |
| `mris_curvature` | 平均曲率和高斯曲率 |
| `mris_place_surface` | 面积、平滑曲率和厚度 |
| `mris_inflate` | 膨胀和沟深 |
| `mris_sphere` | 球面生成 |
| `mris_register` | 基于折叠特征的球面图谱配准 |
| `mri_label2label` | 在不同表面间传递皮层掩膜 |
| `mri_surf2surf` | 分区和标量映射 |

构建器不安装 `mri_annotation2label`、`mris_anatomical_stats`、`mri_convert`、`mris_convert`
或 `recon-all`。Python 统计仍是文档定义的**表面统计表**，不是 recon-all 全部指标的替代。
皮层掩膜的跨表面传递与从分区标注导出脑区标签不同，前者仍需 `mri_label2label`。
Python 标量堆叠也不等于替代 `mri_surf2surf` 中的球面对应映射。

## 可复现性与许可

严格对照时，两组保持相同表面／参考输入、线程设置与 `--sphere-seed 1234`
（主入口为 `--postprocess-sphere-seed 1234`）。默认不设种子的原有行为保留。
不同工具链或平台重新编译后，不保证浮点输出逐位相同。

```bash
python -m postprocessing.core.runtime --runtime ./local-components
python -m postprocessing.core.compare \
  --reference ./reference/<subID> --candidate ./postprocessed/<subID> \
  --hemi left --output ./outputs/surface-comparison.json
```

本地记录源码版本、依赖版本、构建配置及安装文件校验和，使用前验证组件完整性。
不复制或绕过私人激活文件。FreeSurfer 代码／图谱及第三方库保留各自许可，
不会改成 DDSurfer 的 MIT 许可。这里复用原始算法，不是独立重写。

- [固定的上游源码版本](https://github.com/freesurfer/freesurfer/tree/7eb846079b0dc0c92e8313205a3d2387b5c7a354)
- [官方构建说明](https://surfer.nmr.mgh.harvard.edu/fswiki/BuildGuide)
- [FreeSurfer 许可证](../assets/LICENSE.txt)
- [FreeSurfer 激活注册](https://surfer.nmr.mgh.harvard.edu/registration.html)
