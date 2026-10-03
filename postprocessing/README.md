## Surface Postprocessing

Python dependencies are shared with the rest of DDSurfer and defined in the
repository-root [`environment.yml`](../environment.yml) and
[`requirements.txt`](../requirements.txt). See the root
[installation instructions](../README.md#key-dependencies).

This is a plug-and-play surface-only pipeline. Configure FreeSurfer,
`FREESURFER_HOME`, `FS_LICENSE`, and the installed `fsaverage`. The only
required data inputs are four native white/pial cortical surfaces, obtained
by DDSurfer or another reconstruction method:

```bash
bash postprocessing/run.sh --subject <subID> \
  --lh-white ./outputs/<subID>/ddsurfer/lh.white.obj --lh-pial ./outputs/<subID>/ddsurfer/lh.pial.obj \
  --rh-white ./outputs/<subID>/ddsurfer/rh.white.obj --rh-pial ./outputs/<subID>/ddsurfer/rh.pial.obj \
  --output-root ./outputs
```

Set `FREESURFER_HOME` and `FS_LICENSE`. Use `--hemi left` or `right` for one
hemisphere and `--atlases aparc` to select a cortical atlas. Both hemispheres
and the `aparc,aparc.a2009s` atlases are processed by default. The output root
is a FreeSurfer `SUBJECTS_DIR`, with a link to the installed `fsaverage`.

Results are written directly into `outputs/<subID>/`, using standard
`surf/`, `label/`, `stats/`, `fsaverage/`, `mri/`, and `logs/` directories.
The main DDSurfer pipeline disables this stage by default; `--post-process`
enables it after native conversion.

Inputs must be closed triangular surfaces in native scanner RAS millimetres.
Each white/pial pair must have matching vertex indices and faces. Indexed
OBJ/PLY/OFF and FreeSurfer binary surfaces with valid volume geometry are supported.
Paired STL files must retain corresponding triangle order; both are welded
using the same vertex mapping. Other triangle orders are rejected, not guessed.

DWI, DTI and structural MRI are not required. A geometry-only LIA reference
is constructed from the surfaces for FreeSurfer coordinate bookkeeping. It
contains no anatomical intensities and is marked `geometry_only` in QC logs.
Optionally pass `--brain-source` with a native scalar MRI. Such an MRI is
reoriented to LIA while retaining native-world coordinates. Axis-aligned inputs
use lossless permutations/flips; oblique inputs are resliced to an axis-aligned
native grid. No MRI template registration is applied in this step.
Scanner RAS is converted to FreeSurfer surface RAS using the complete MRI
affine and `vox2ras_tkr`; this is a coordinate conversion, not another MRI
registration. See the [FreeSurfer coordinate conventions](https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems).

A successful coordinate round trip alone does not verify the anatomical axes
needed by sphere registration. The LIA reference makes surface RAS and native
scanner RAS share anatomical axis directions; their origins can still differ.

Outputs include curvature, sulcal depth, inflated/spherical surfaces,
[FreeSurfer thickness](https://surfer.nmr.mgh.harvard.edu/fswiki/mris_thickness),
surface areas, atlas annotations/labels, regional statistics and `fsaverage`
overlays. Thickness follows FreeSurfer's default 5 mm limit. No segmentation
volume is inferred; voxel-wise tissue volumes are not reported.

Regional `.stats` tables contain vertex counts, white/pial surface areas,
thickness mean/standard deviation, and mean absolute curvature. Only vertices
inside the transferred cortex label are included. They are surface-only tables,
not the complete `recon-all` statistics: `GrayVol`, eTIV, and segmentation-derived
volumes are intentionally absent. The calculation uses annotation indices,
not RGB colour-table entries, and does not need `wm.mgz` or `aseg.mgz`.

Commands and coordinate checks are saved under `<subID>/logs/`. Failed commands
stop the pipeline. Re-running with the same inputs resumes completed stages;
changed inputs or settings require a new output directory.

The implementation follows the standalone curvature, area and thickness
commands in [FastSurfer recon-surf](https://github.com/Deep-MI/FastSurfer/blob/dev/recon_surf/recon-surf.sh)
when `mris_place_surface` is available, with a fallback for older FreeSurfer
installations. These commands measure the supplied surfaces; they do not
reposition them. Hemispheres run in parallel, sharing the total `--threads`
budget; use `--serial` to disable this. Logs are separated by hemisphere.

Sphere and atlas registration are retained for atlas transfer and cross-subject
analysis. FastSurfer's spectral `qsphere` accelerates its topology-fixing stage;
it is not substituted for the final registration sphere here. Its segmentation-
guided DKT transfer also requires segmentation labels not produced by DDSurfer.
