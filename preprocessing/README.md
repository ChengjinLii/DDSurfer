# Preprocessing

This directory contains the complete volume-preprocessing workflow.
`run.sh` calls `dti.sh` to estimate DTI and register scalar maps, then masks,
resamples and z-score normalizes them for surface inference.

The main pipeline selects `--minimal`: FA, MD, and the three eigenvalue maps.
Unused Trace and DTI-cache `NormMasked` maps are not computed in this mode.
Standalone `run.sh` and `dti.sh` retain the complete scalar outputs by default.
Use `--minimal` explicitly for inference-only preprocessing.

Independent DTI scalar and resampling commands can run concurrently with
`--jobs 2` (default: 1), or `--preprocess-jobs 2` on the main pipeline.
Tensor estimation and atlas registration remain ordered, with unchanged
commands, interpolation and thread settings. Limit concurrency to available
memory and CPU resources.

Stages record input/output checksums and publish complete results atomically.
Detached NRRD data files are checked together with their headers. The final
normalized images are separate from unnormalized intermediates, so a resumed
run does not normalize an already-normalized image again. Masks are shared
within one Python process; both resampling steps and their file precision are
retained. Intermediate files and receipts stay inside the disposable cache.

The image helpers `mask.py`, `resample.py`, and `normalize.py` are kept in this
directory. `normalize_dti.py` handles DTI-cache normalization separately;
`normalize.py` performs the final per-volume z-score normalization used by
surface inference. Their mask and background handling are kept distinct.

Install Python dependencies using the repository-root
[`environment.yml`](../environment.yml) or [`requirements.txt`](../requirements.txt).
Slicer with SlicerDMRI is required separately; see the root
[installation instructions](../README.md#key-dependencies).

The main DDSurfer pipeline invokes this stage automatically. To run volume
preprocessing alone:

```bash
bash preprocessing/run.sh --subject <subID> --raw-input-root ./inputs
```

It covers:

- DWI NIfTI to NHDR conversion
- tensor estimation and b0 extraction
- scalar-map generation
- rigid/affine registration to the atlas reference image
- scalar-map resampling into atlas space
- optional scalar normalization inside the registered mask

DTI-only entry point:

- `dti.sh`

Required inputs are a corrected 4D DWI NIfTI, its bval/bvec gradient table,
and a 3D brain mask in the same native voxel grid and physical space. Gradient
correction and DWI motion/eddy/susceptibility correction must already be done.
DTI is computed internally, following the input-to-tensor workflow in
[DDParcel](https://github.com/zhangfanmark/DDParcel/blob/main/process.sh).
Tensor fitting retains the DDSurfer WLS configuration.

Example with explicit input files (no HCP directory structure required):

```bash
bash preprocessing/dti.sh --subject <subID> \
  --dwi ./inputs/<subID>/dwi.nii.gz --bval ./inputs/<subID>/dwi.bval \
  --bvec ./inputs/<subID>/dwi.bvec --mask ./inputs/<subID>/mask.nii.gz \
  --output-root ./outputs/<subID>/.cache/dti
```

Directory discovery also accepts DDParcel-style files directly under the
input root or `<input_root>/<subject_id>/`. The HCP layout remains supported:

- `<input_root>/<subject_id>/T1w/Diffusion/data.nii.gz`
- `<input_root>/<subject_id>/T1w/Diffusion/bvals`
- `<input_root>/<subject_id>/T1w/Diffusion/bvecs`
- `<input_root>/<subject_id>/T1w/Diffusion/nodif_brain_mask.nii.gz`

Within the main pipeline, generated files are cached under
`outputs/<subject_id>/.cache/dti/<subject_id>/`:

- `<subject_id>-dti-*-Reg.nii.gz`
- `<subject_id>-mask-Reg.nii.gz`
- `<subject_id>-b0ToAtlasT2.tfm`
- `<subject_id>-b0.nhdr` and its detached payload

`inputs.py` checks image geometry, gradient counts and tensor-design rank.
It records input checksums before estimation. Identical inputs may resume;
different inputs or older unrecorded outputs require a new output directory.
Input NIfTI files are not modified.

The main pipeline removes this cache after exporting native surfaces, unless
`--keep-cache` is selected. Raw-input records and the native transform are
retained under `outputs/<subject_id>/logs/`.

Directory-discovery example:

```bash
bash preprocessing/dti.sh \
  --subject <subID> \
  --input-root ./inputs \
  --output-root ./outputs/<subID>/.cache/dti
```
