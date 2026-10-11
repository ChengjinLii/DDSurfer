# Postprocessing Optimizations

**Less repeated work, with the surface geometry and measurement definitions preserved.**

[Usage](README.md) | [Python Replacements](#python-replacements) | [Retained Algorithms](#retained-algorithms) | [Validation](#validation) | [中文](OPTIMIZATIONS.zh-CN.md)

---

## Python Replacements

| Operation | Implementation | Scope |
| --- | --- | --- |
| Annotation to regional labels | [`labels.py`](labels.py) replaces `mri_annotation2label`. | Uses annotation color codes and white-surface coordinates; preserves vertex order, surface RAS, text precision and present `unknown` labels. Does not change parcellation. |
| Regional statistics | [`stats.py`](stats.py) is called in-process instead of starting another Python interpreter. | Same surface-only table writer and definitions; no new measurements. |
| Shared measurement input | [`pipeline.py`](pipeline.py) creates a per-hemisphere input cache only when rebuilding statistics or labels. | Reuses white vertices and cortex/measurement arrays across atlases; shares relevant measurements with overlay packing and the thickness check. Original dtype and reduction operations are retained. |
| Label text writing | [`labels.py`](labels.py) writes at most 4,096 rows per text block. | Avoids the full five-column intermediate array and generic per-row `numpy.savetxt` loop while preserving text precision and row order. |
| Surface/reference conversion | Existing [`geometry.py`](geometry.py) uses nibabel, NumPy and trimesh. | Reads supported meshes, prepares the FreeSurfer reference, converts scanner RAS to surface RAS, and checks topology and coordinate round trips without conversion subprocesses. |
| Area on older installations | Existing NumPy vertex-area fallback. | Used only when `mris_place_surface` is unavailable; the modern FreeSurfer area path remains unchanged. |

The regional tables are **not a replacement for full `mris_anatomical_stats`
or `recon-all` results**. They contain vertex counts, white/pial area,
thickness mean/std and mean absolute curvature within the cortex label.
They do not estimate `GrayVol`, eTIV or segmentation-derived tissue volumes.

---

## Shared Mapping and Reliable Reuse

- **One mapping instead of four.** [`overlays.py`](overlays.py) stacks thickness,
  curvature, sulcal depth and pial area. One `mri_surf2surf` command maps all four
  frames using the same registration and mapping defaults as before.
- **FreeSurfer still writes the outputs.** Its `--split` option writes the scalar
  MGH files. Python validates all four before copying them to staged output paths,
  without dropping FreeSurfer's additional MGH metadata.
- **Bounded bilateral parallelism.** The existing scheduler splits the total
  thread budget between hemispheres rather than assigning the full budget twice.
- **Content-checked resume.** Input, implementation and output checksums govern
  reuse. Missing/corrupted labels are rebuilt in Python; missing/corrupted overlays
  trigger one shared mapping. Python stages validate temporary outputs before
  publishing files and recording success. Interrupted publication has no valid
  completion receipt and is retried.
- **Lazy older-installation fallback.** Raw white/pial mesh pairs are loaded
  only inside an area-rebuild callback, not before checking completed stages.
  The existing mean-curvature fallback copies are now checksum-verified Python
  stages rather than unconditional file rewrites. The fallback area formula and
  curvature values are unchanged.

Shared input arrays exist only for the current hemisphere's processing pass,
not as a persistent subject cache. They trade a modest amount of temporary CPU
memory for fewer file reads/decodes; no lower peak-memory claim is made. Completed
statistics do not trigger eager measurement loading on resume.

The per-subject lock and existing spatial checks remain in place. Input surfaces
are read, not modified. No new dependency or command-line option is required.

---

## Retained Algorithms

| Operation | FreeSurfer command |
| --- | --- |
| White/pial area and smoothed curvature | `mris_place_surface --area-map` / `--curv-map` |
| Mean/Gaussian curvature | `mris_curvature` |
| Inflation and sulcal depth | `mris_inflate` |
| Sphere generation and registration | `mris_sphere`, `mris_register` |
| Thickness | `mris_place_surface --thickness`; `mris_thickness` fallback |
| Cortex label transfer | `mri_label2label` |
| Atlas and overlay transfer | `mri_surf2surf` |

No nearest-neighbour Python substitute is used for cortex or atlas transfer.
No registration iterations, smoothing settings or thickness limits are reduced.
A NumPy area implementation was checked against the modern FreeSurfer output:
differences reached about `4.8e-7 mm^2` per vertex in the checked case. Small as
they are, the modern area command is retained rather than introducing a numerical
change solely to remove a short subprocess.

Alternative sphere/registration algorithms and custom cortex-transfer routines
are not substituted: output equivalence or improved atlas accuracy has not been
demonstrated. The additional optimizations change data handling, not these
algorithms.

---

## Validation

Real-command checks used **FreeSurfer 7.4.1**, subject `100307` left-hemisphere
results, and the installed fsaverage labels. The source results were unchanged.

| Check | Result |
| --- | --- |
| Regional label export | 219 files across `aparc` and `aparc.a2009s`, including left/right fsaverage cases: same file sets and byte-identical label files. |
| Four fsaverage overlays | Byte-identical MGH files against four separate mapping commands: values, shape, affine, headers and additional metadata preserved. |
| Regional statistics | Byte-identical to the existing surface-only table writer. |
| Shared-input summary pass | 113 label/statistics/packed-overlay files across two atlases were byte-identical to the preceding implementation; the four mapped MGH files were rechecked against separate FreeSurfer calls. |
| CPU regression suite | 153 tests pass, including environment diagnostics, bilateral scheduling, shared input reads, modern/legacy metric resume, malformed input and missing/corrupted output checks. |

Representative stage timings with a two-thread FreeSurfer budget:

| Stage | Before | After |
| --- | --- | --- |
| Export 35 `aparc` labels | 1.13 s | 0.37 s |
| Map four fsaverage overlays | 13.68 s | 3.23 s |

Further Python-stage comparisons against the preceding implementation used three
warm-file runs on the same hemisphere. Both versions retained the same numeric
operations and output formats:

| Operation | Before | After |
| --- | --- | --- |
| Export 35 labels, including white-mesh reading | 0.35 s | 0.26 s |
| Two-atlas label/statistics export, overlay packing and thickness check | 0.88 s | 0.65 s |
| Python scalar-map reads in that summary pass | 15 | 6 |
| White-mesh / cortex-label reads in that summary pass | 2 / 2 | 1 / 1 |

These counts measure data-reader calls, not all filesystem reads; checksum
verification is still performed. Tests also confirm that fully reusable legacy
metric outputs require no white/pial mesh loading or unchanged curvature rewrites.

These are local measurements, **not whole-pipeline speedup guarantees**. Sphere
generation and registration remain the main cost and were not changed. Changed
stages were checked with real FreeSurfer commands; the complete postprocessing
workflow was regression-tested with command fixtures, not rerun end to end for
this timing comparison. Other FreeSurfer versions have not been benchmarked.

Run the CPU checks from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' python3 -B -m unittest discover -s tests -q
```

### References

The implementation was checked against the installed FreeSurfer source revision:
[annotation-to-label export](https://github.com/freesurfer/freesurfer/blob/7eb8460/mri_annotation2label/mri_annotation2label.cpp),
[label format writer](https://github.com/freesurfer/freesurfer/blob/7eb8460/utils/label.cpp),
and [multi-frame surface mapping](https://github.com/freesurfer/freesurfer/blob/7eb8460/mri_surf2surf/mri_surf2surf.cpp).
