<a id="surface-utilities"></a>

# 皮层表面工具

**独立使用的格式转换与平移工具。**

[DDSurfer](../README.zh-CN.md) | [后处理](../postprocessing/README.zh-CN.md) | [English](README.md)

这些工具不属于主推理流程。主流程直接输出 OBJ 表面，无需额外格式转换。

---

<a id="available-tools"></a>

## 可用工具

| 脚本 | 用途 | 参数 |
| --- | --- | --- |
| `obj_to_stl.py` | 将 OBJ 网格导出为 STL。 | `--input_obj`、`--output_stl` |
| `stl_to_obj.py` | 将 STL 网格导出为 OBJ。 | `--input_stl`、`--output_obj` |
| `translate_mesh.py` | 施加固定平移 `[85, 132, 70]`。 | `--input_obj`、`--output_obj` |

<a id="usage"></a>

## 使用方法

在仓库根目录运行：

```bash
python3 tools/obj_to_stl.py --input_obj ./surface.obj --output_stl ./surface.stl
python3 tools/stl_to_obj.py --input_stl ./surface.stl --output_obj ./surface.obj
```

通过 `python3 tools/<script>.py --help` 查看完整参数。

---

<a id="before-converting"></a>

## 转换前的注意事项

- 格式转换可能改变顶点索引。配合使用白质与软脑膜网格前，应检查两者的顶点对应关系。
- STL 不保存顶点索引。向 [后处理](../postprocessing/README.zh-CN.md#surface-requirements)
  提供配对 STL 文件时，应保留相互对应的三角面片顺序。
- 固定平移工具**不是配准工具**，也不是依据仿射矩阵进行的坐标转换。
  请勿将其用于当前 DDSurfer 的输出。
