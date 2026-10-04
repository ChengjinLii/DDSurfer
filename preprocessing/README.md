# Preprocessing

**From corrected diffusion MRI to surface-inference inputs.**

[DDSurfer](../README.md) | [Run](#run) | [Brain Masking](#optional-brain-masking) | [Outputs](#outputs)

The main DDSurfer pipeline runs this stage automatically. `run.sh` estimates
DTI through `dti.sh`, then prepares the registered volumes for inference.

---

## Requirements

| Input | Requirement |
| --- | --- |
| DWI | Corrected 4D NIfTI with b0 and diffusion-weighted volumes. |
| bval / bvec | Gradient table matching the DWI volume count; bvecs may be 3xN or Nx3. |
| Brain mask | Nonempty 3D NIfTI on the DWI's native grid and physical space. |

Motion, eddy-current and susceptibility corrections, including gradient
updates, must already be complete. A scalar map or fitted tensor is not a raw
DWI input. Source files are never modified.

Install Python dependencies from the root [`environment.yml`](../environment.yml)
or [`requirements.txt`](../requirements.txt). Slicer with SlicerDMRI is required
separately. See [installation](../README.md#key-dependencies).

---

## Run

### Complete Volume Preprocessing

```bash
bash preprocessing/run.sh --subject <subID> --raw-input-root ./inputs
```

### DTI Estimation and Registration Only

```bash
bash preprocessing/dti.sh --subject <subID> \
  --dwi ./inputs/<subID>/dwi.nii.gz \
  --bval ./inputs/<subID>/dwi.bval \
  --bvec ./inputs/<subID>/dwi.bvec \
  --mask ./inputs/<subID>/mask.nii.gz \
  --output-root ./outputs/<subID>/.cache/dti
```

Directory discovery accepts the default `inputs/<subID>/` bundle, subject-named
files, and HCP's `T1w/Diffusion/` layout. Use explicit paths for other layouts.
Run either entry point with `--help` for its options and standalone output paths.

### Workflow

| Step | Operation |
| --- | --- |
| 1 | Validate image geometry, gradient counts and tensor-design rank. |
| 2 | Convert the DWI to NHDR; fit the tensor with WLS and extract b0. |
| 3 | Calculate FA, MD, eigenvalues and Trace in native space. |
| 4 | Register b0 to the atlas and apply the same transform to scalar maps. |
| 5 | Mask, resample and z-score normalize inference inputs. |

`--minimal` prepares only the five inference channels in atlas space and skips
unused normalized DTI-cache maps. Native Trace is still calculated. The main
pipeline selects this mode; standalone scripts retain full scalar outputs by
default.

---

## Optional Brain Masking

Automatic masking is **off by default**. In the main pipeline, `--auto-mask`
allows the mask to be omitted:

```bash
python3 run_ddsurfer_pipeline.py --subject <subID> --auto-mask
```

The method averages DWI volumes with `b <= 50` and runs FreeSurfer
`mri_synthstrip` on that native b0 image. It follows the mean-b0 masking approach
described in the [MRtrix3 masking guide](https://userdocs.mrtrix.org/en/dev/dwi_preprocessing/masking.html).
MRtrix3 is not a dependency of this implementation.

**Setup and behavior**

- Configure FreeSurfer with `mri_synthstrip` and its model weights.
- Set `FREESURFER_HOME` or pass `--freesurfer-home <path>`.
- Generation runs on CPU; `--mask-threads` defaults to 4.
- A supplied mask takes priority and is never replaced.
- The generated binary mask must match the DWI's shape and physical affine.
- Inspect the mask for anatomical quality; geometry checks alone do not ensure it.

The generated mask is retained as `outputs/<subID>/dti/<subID>-brainmask.nii.gz`.
`logs/mask.json` records its inputs, method and checksums; unchanged inputs can
reuse it even after intermediate cache cleanup. Changed inputs or an edited
generated mask require a new output directory.

To prepare a mask independently, then pass it to `run.sh` or `dti.sh`:

```bash
python3 preprocessing/brain_mask.py \
  --dwi ./inputs/<subID>/dwi.nii.gz \
  --bval ./inputs/<subID>/dwi.bval \
  --bvec ./inputs/<subID>/dwi.bvec \
  --output ./outputs/<subID>/dti/<subID>-brainmask.nii.gz \
  --work-dir ./outputs/<subID>/.cache/mask \
  --record ./outputs/<subID>/logs/mask.json
```

No T1 image or `recon-all` run is needed for this option. Automatic masks can
differ from dataset-provided masks and therefore can affect downstream results.

---

## Outputs

| Location in the main pipeline | Contents | Retained by default |
| --- | --- | --- |
| `outputs/<subID>/dti/` | Native FA, MD, three eigenvalue maps and Trace; optional generated mask. | Yes |
| `outputs/<subID>/logs/` | Input records, native-DTI summary and registration transform. | Yes |
| `outputs/<subID>/.cache/dti/` | Tensor, b0 and registered scalar maps. | No |
| `outputs/<subID>/.cache/volumes/` | Masked, normalized inference volumes. | No |

`export.py` converts native scalar maps to NIfTI without resampling, masking or
normalization, checking them against the original DWI geometry. Older retained
caches without Trace can export the five available maps.

The atlas-to-native transform is preserved as `logs/b0ToAtlasT2.tfm`.
Intermediate files are removed after a successful main-pipeline run unless
`--keep-cache` is set. Failed runs retain their cache for diagnosis.

---

## Performance and Reuse

- Use `--jobs 2` for independent DTI stages, or `--preprocess-jobs 2` on the main
  pipeline. The default is 1; limit concurrency to available CPU and memory.
- Tensor fitting and registration stay ordered. Interpolation, registration
  settings and both resampling steps are unchanged.
- Completed stages are reused only after input/output checksum validation;
  atomic writes prevent partially written files being treated as complete.
- Normalized outputs are separate from unnormalized intermediates, preventing
  repeated normalization on resume.
- Different raw inputs, or unrecorded older outputs, require a new output
  directory rather than silently reusing unrelated results.

### Helpers

| Files | Purpose |
| --- | --- |
| `inputs.py`, `brain_mask.py` | Input validation and optional brain masking. |
| `volumes.py`, `mask.py`, `resample.py` | Masking and inference-grid resampling. |
| `normalize.py`, `normalize_dti.py` | Final z-score normalization and separate DTI-cache normalization. |
| `export.py`, `cache.py` | Native scalar export and validated stage reuse. |

The two normalization helpers serve different stages; their mask and background
handling are not interchangeable.
