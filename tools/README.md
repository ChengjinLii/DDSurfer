# Surface Utilities

**Standalone format-conversion and translation helpers.**

[DDSurfer](../README.md) | [Postprocessing](../postprocessing/README.md)

These tools are not part of the main inference pipeline, which writes OBJ
surfaces directly.

---

## Available Tools

| Script | Purpose | Arguments |
| --- | --- | --- |
| `obj_to_stl.py` | Export an OBJ mesh as STL. | `--input_obj`, `--output_stl` |
| `stl_to_obj.py` | Export an STL mesh as OBJ. | `--input_stl`, `--output_obj` |
| `translate_mesh.py` | Apply the fixed translation `[85, 132, 70]`. | `--input_obj`, `--output_obj` |

## Usage

Run from the repository root:

```bash
python3 tools/obj_to_stl.py --input_obj ./surface.obj --output_stl ./surface.stl
python3 tools/stl_to_obj.py --input_stl ./surface.stl --output_obj ./surface.obj
```

Use `python3 tools/<script>.py --help` for all arguments.

---

## Before Converting

- Format conversion can change vertex indices. Check white/pial correspondence
  before using converted meshes together.
- STL does not store indexed vertices. Keep corresponding triangle order when
  providing paired STL files to [postprocessing](../postprocessing/README.md#surface-requirements).
- The fixed-translation helper is **not registration** or an affine-aware
  coordinate conversion. Do not apply it to current DDSurfer outputs.
