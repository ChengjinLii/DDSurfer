# Surface Postprocessing

**Plug-and-play morphometry and cortical parcellation from four surfaces.**

[DDSurfer](../README.md) | [Run](#run) | [Surface Requirements](#surface-requirements) | [Outputs](#outputs)

Use surfaces from DDSurfer or another reconstruction method. DWI, DTI and
structural MRI are not required. The main pipeline disables postprocessing by
default; `--post-process` enables it after native-space surface conversion.

---

## Requirements

| Component | Setup |
| --- | --- |
| FreeSurfer | Configure `FREESURFER_HOME`, `FS_LICENSE` and the installed `fsaverage`. |
| Python packages | Shared root [`environment.yml`](../environment.yml) or [`requirements.txt`](../requirements.txt). |
| Surfaces | Left/right white and pial, with matching white/pial topology per hemisphere. |

See the root [installation instructions](../README.md#key-dependencies).

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

## Reuse and Execution

Completed stages resume after checksum validation. Changed inputs or settings
require a new output directory. Upstream dependencies are checked alongside
outputs, and command failures stop the pipeline rather than publishing success.

Both hemispheres share the total thread budget and have separate command logs.
Use `--serial` where concurrent execution is undesirable.

The measurement commands follow [FastSurfer recon-surf](https://github.com/Deep-MI/FastSurfer/blob/dev/recon_surf/recon-surf.sh)
when `mris_place_surface` is available, with a fallback for older FreeSurfer.
Sphere registration remains the atlas-transfer step; segmentation-guided
parcellation is not substituted because no tissue segmentation is required here.
