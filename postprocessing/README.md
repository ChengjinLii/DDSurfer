# Surface Postprocessing

**Plug-and-play morphometry and cortical parcellation from four surfaces.**

[DDSurfer](../README.md) | [Workflow](#workflow) | [Run](#run) | [Surface Requirements](#surface-requirements) | [Outputs](#outputs) | [Optimizations](#optimizations-over-freesurfer) | [中文](README.zh-CN.md)

Use surfaces from DDSurfer or another reconstruction method. DWI, DTI and
structural MRI are not required. The main pipeline disables postprocessing by
default; `--post-process` enables it after native-space surface conversion.

---

## Workflow

Postprocessing starts from **native-space white and pial surfaces**, after
DDSurfer's MNI-to-native conversion, or from equivalent surfaces produced by
another method. It does not rerun surface reconstruction or register the anatomy
to MNI. Spherical registration below is for surface-based atlas transfer.

```text
Native white/pial surfaces
  -> Reference preparation and scanner-RAS to surface-RAS conversion
  -> Surface area and curvature
  -> Inflation and sulcal depth
  -> Sphere generation and spherical atlas registration
  -> Thickness, cortex label and cortical parcellations
  -> Regional labels/statistics and fsaverage overlays
```

| Stage | Processing | Main results |
| --- | --- | --- |
| Prepare | Python checks white/pial correspondence, prepares an LIA reference and converts coordinates. | `mri/brain.mgz`, `mri/orig.mgz`, FreeSurfer white/pial surfaces and spatial QC. |
| Measure | FreeSurfer computes area, mean/Gaussian curvature and smoothed curvature on white/pial surfaces. | `area`, `area.pial`, `white.H/K`, `pial.H/K`, `curv`, `curv.pial`. |
| Inflate | `mris_inflate`, followed by curvature measurement on the inflated surface. | `inflated`, `sulc`, `inflated.H/K`. |
| Register | `mris_sphere` and `mris_register` establish spherical correspondence with the atlas. | `sphere`, `sphere.reg`. |
| Transfer | FreeSurfer measures thickness and transfers the fsaverage cortex label and selected annotations. | `thickness`, `cortex.label`, `aparc.annot`, `aparc.a2009s.annot`. |
| Summarize | Python exports regional labels/statistics; one FreeSurfer command maps four scalar overlays. | Regional `.label` and `.stats` files, plus fsaverage `.mgh` overlays. |

Both hemispheres follow this sequence, in parallel within the total thread
budget unless `--serial` is selected. Native white/pial anatomy is preserved;
inflated and spherical surfaces are additional representations, not replacements
for the supplied surfaces.

Measurement commands follow [FastSurfer recon-surf](https://github.com/Deep-MI/FastSurfer/blob/dev/recon_surf/recon-surf.sh)
when `mris_place_surface` is available, with a fallback for older FreeSurfer.
The selected atlas labels are transferred through sphere registration rather
than inferred from a tissue segmentation.

---

## Requirements

| Component | Setup |
| --- | --- |
| FreeSurfer | Configure `FREESURFER_HOME`, `FS_LICENSE` and the installed `fsaverage`. |
| Python packages | Shared root [`environment.yml`](../environment.yml) or [`requirements.txt`](../requirements.txt). |
| Surfaces | Left/right white and pial, with matching white/pial topology per hemisphere. |

See the root [installation instructions](../README.md#key-dependencies).

The main pipeline checks the required FreeSurfer executables, license and atlas
files before preprocessing. These checks are not required for surface inference
without `--post-process`. Final-result reuse also verifies postprocessing outputs.

---

## Run

### With DDSurfer

```bash
python3 run_ddsurfer_pipeline.py --subject <subID> --post-process
```

### Independently

```bash
bash postprocessing/run.sh --subject <subID> \
  --lh-white ./outputs/<subID>/ddsurfer/lh.white.obj \
  --lh-pial ./outputs/<subID>/ddsurfer/lh.pial.obj \
  --rh-white ./outputs/<subID>/ddsurfer/rh.white.obj \
  --rh-pial ./outputs/<subID>/ddsurfer/rh.pial.obj \
  --output-root ./outputs
```

| Standalone option | Behavior |
| --- | --- |
| `--hemi left`, `right` or `both` | Select hemispheres; default: both. |
| `--atlases aparc` | Select atlas names; default: `aparc,aparc.a2009s`. |
| `--brain-source <file>` | Optional native scalar MRI for the FreeSurfer reference. |
| `--threads 4` | Total thread budget, shared across hemispheres. |
| `--serial` | Process hemispheres sequentially instead of in parallel. |
| `--freesurfer-home <path>` | Override `FREESURFER_HOME`. |

The main pipeline exposes the corresponding `--postprocess-hemi`,
`--postprocess-atlases`, `--postprocess-threads` and `--postprocess-serial` options.
Its reference image defaults to the native b0 produced during DTI estimation.

---

## Surface Requirements

Surfaces must be **closed triangular meshes in native scanner-RAS millimetres**.
Each white/pial pair must share vertex indices and faces.

| Format | Notes |
| --- | --- |
| OBJ / PLY / OFF | Indexed surfaces; preserve white/pial correspondence. |
| FreeSurfer binary | Must include valid volume geometry. |
| STL | Paired files must retain corresponding triangle order; a shared welding map reconstructs indices. |

Invalid correspondence is rejected, not guessed through nearest neighbours.
DDSurfer's default surface format is OBJ.

### Coordinate Handling

Without an MRI, a geometry-only LIA reference is constructed from the surfaces.
It contains no anatomical intensities and is marked `geometry_only` in QC logs.
With `--brain-source`, native image axes are aligned to LIA: permutations/flips
for axis-aligned inputs, reslicing for oblique inputs. Neither path registers
the image to a template.

Scanner RAS is converted to FreeSurfer surface RAS using the MRI affine and
`vox2ras_tkr`. This changes the coordinate representation, not the anatomical
location. The LIA reference preserves the anatomical axis convention required
for sphere registration; a coordinate round trip alone cannot verify that
convention. See [FreeSurfer coordinates](https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems).

---

## Outputs

Results are written directly under `outputs/<subID>/`:

| Directory | Representative files | Purpose |
| --- | --- | --- |
| `surf/` | `lh.white`, `lh.pial`, `lh.curv`, `lh.sulc`, `lh.thickness`, `lh.area` | Native surfaces and morphometry. |
| `surf/` | `lh.inflated`, `lh.sphere`, `lh.sphere.reg` | Inflation and spherical atlas registration. |
| `label/` | `lh.cortex.label`, `lh.aparc.annot`, `lh.aparc.a2009s.annot` | Cortex mask, parcellations and regional labels. |
| `stats/` | `lh.aparc.stats`, `lh.aparc.a2009s.stats` | Regional surface statistics. |
| `fsaverage/` | `lh.thickness.mgh`, `lh.curv.mgh`, `lh.sulc.mgh`, `lh.area.pial.mgh` | Overlays transferred to the standard surface. |
| `mri/` | `brain.mgz`, `orig.mgz` | FreeSurfer reference volumes. |
| `logs/` | Command logs, timing and coordinate checks. | Execution records and QC. |

Right-hemisphere files use `rh.`. Atlas and hemisphere selections determine
which files are generated. The output root acts as FreeSurfer `SUBJECTS_DIR`
and also contains a link to the installed `fsaverage` template; this is separate
from the subject's `fsaverage/` overlay directory.

### Interpreting the Measurements

- Curvature, area and thickness measure the supplied surfaces; the measurement
  commands do not reposition them.
- Thickness uses FreeSurfer's default 5 mm limit.
- Regional tables report vertex counts, white/pial area, thickness mean/standard
  deviation and mean absolute curvature within the transferred cortex label.
- These are **surface-only statistics**, not full `recon-all` tables: no `GrayVol`,
  eTIV or segmentation-derived tissue volumes are inferred.
- A geometry-only `mri/` reference is not a reconstructed structural image.

---

## Optimizations over FreeSurfer

DDSurfer retains FreeSurfer's core surface algorithms while reducing repeated
work around them. Some improvements replace individual commands; others improve
task scheduling and reuse. This is a **surface-only analysis workflow**, not a
faster equivalent of every output produced by a full `recon-all` run.

| Improvement | Implementation | Benefit |
| --- | --- | --- |
| Python regional label export | Replaces `mri_annotation2label` with annotation-based vertex selection and the same label text format. | Removes a FreeSurfer subprocess; 35-label export took **1.13 s -> 0.37 s** in the local comparison. |
| Shared fsaverage mapping | Packs thickness, curvature, sulcal depth and pial area into one `mri_surf2surf --split` call rather than four calls. | Shares sphere loading and correspondence computation; the checked stage took **13.68 s -> 3.23 s**, about **76% less time**. FreeSurfer still writes the complete MGH metadata. |
| In-process statistics | Calls the existing Python surface-only table writer directly. | Avoids starting and importing dependencies in another Python interpreter; measurement definitions stay unchanged. |
| Shared per-hemisphere inputs | Lazily reuses white vertices, cortex indices and measurement arrays across atlases, overlay packing and thickness checks. | In the default two-atlas summary phase, Python scalar-map reads fall from **15 to 6**; white mesh and cortex-label reads each fall from **2 to 1**. |
| Bounded label-text blocks | Writes up to 4,096 label rows per block without constructing a full five-column text-export array. | Preserves the same format; 35-label writer time fell from **0.35 s to 0.26 s** against the preceding Python writer, with geometry reading included in both. |
| Python geometry preparation | Converts formats/coordinates and prepares the reference without separate conversion commands. | Avoids conversion subprocesses and adds explicit topology, field-of-view and coordinate-round-trip checks. |
| Bounded bilateral parallelism | Shares the total thread budget across hemispheres. | Allows independent work to overlap without doubling the configured thread budget. |
| Checksum-based resume and locking | Reuses verified stages; validates staged Python outputs before publication. | Avoids repeated work on reruns, prevents duplicate subject processing, and retries incomplete/corrupted outputs rather than accepting them as complete. |
| Lazy legacy-metric fallback | Loads a mesh pair only when an older-installation area output needs rebuilding; verifies curvature-copy outputs before reuse. | Completed fallback stages no longer reload white/pial meshes or rewrite unchanged curvature files on each resume. |

Using already reconstructed surfaces also avoids upstream voxel segmentation
and surface reconstruction. That reduces the amount of work needed for this
task, but does not provide segmentation-derived measurements such as `GrayVol`
or eTIV. The NumPy vertex-area fallback is used only on older installations;
modern installations retain FreeSurfer's area computation.

**Unchanged:** curvature, thickness, inflation, sphere generation/registration,
cortex label transfer and atlas mapping retain their FreeSurfer algorithms and
parameters. No iteration count, smoothing setting or thickness limit is reduced
for speed.

The latest shared-reading and block-writing changes reduced the checked
two-atlas Python summary phase from **0.88 s to 0.65 s** (about **26% less time**,
median of three warm-file runs). This phase covers label export, regional
statistics, overlay packing and the thickness check, not FreeSurfer registration
or mapping. Reading counts refer to Python data readers, not all filesystem
activity such as checksum verification.

### Verified Improvements and Limits

- **Equivalent outputs in the checked cases:** 219 regional label files and all
  four fsaverage MGH overlays were byte-identical to the corresponding separate
  FreeSurfer commands. Regional statistics were byte-identical to the existing
  surface-only Python writer.
- **Shared-input checks:** a further two-atlas pass produced 113 byte-identical
  label, statistics and packed-overlay files against the preceding implementation.
  The four mapped MGH files were also rechecked against separate FreeSurfer calls.
- **Measured scope:** FreeSurfer 7.4.1, subject `100307` left-hemisphere results
  and fsaverage label cases, with a two-thread FreeSurfer budget. The timings are
  stage-level measurements, not a whole-pipeline speedup guarantee.
- **Remaining cost:** sphere generation and registration are unchanged and
  remain the main bottleneck. Peak memory reduction has not been measured;
  batched mapping holds four scalar fields together, and shared reading retains
  a small, temporary set of CPU arrays per hemisphere. It is not a persistent
  preload of multiple subjects.
- **Regression coverage:** 153 CPU tests passed, including scheduling, shared
  reads, modern/legacy metric resume and invalid/missing output handling.
  Changed stages were tested with real
  FreeSurfer commands; end-to-end orchestration was tested with command fixtures.

See [implementation details and equivalence checks](OPTIMIZATIONS.md) for source
references and the regression-test command.

---

## Reuse and Execution

Completed stages resume after checksum validation. Changed inputs or settings
require a new output directory. Upstream dependencies are checked alongside
outputs, and command failures stop the pipeline rather than publishing success.

Both hemispheres share the total thread budget and have separate command logs.
Use `--serial` where concurrent execution is undesirable.
