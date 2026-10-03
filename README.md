## DDSurfer Release

DDSurfer reconstructs white and pial cortical surfaces from diffusion MRI inputs.
This release bundles preprocessing utilities, dual-stream TANet inference, and
postprocessing tools in a single repository.

Model weights are stored directly in `ckpts/` and loaded automatically.

---

## Publication

DDSurfer has been accepted and published online in **Advanced Science**:

> Chengjin Li, Wei Zhang, Xi Zhu, Yuqian Chen, Nir A. Sochen, Jarrett Rushmore,
> Carl-Fredrik Westin, Yogesh Rathi, Lauren J. O'Donnell, Ofer Pasternak, and
> Fan Zhang. "DDSurfer: A Weakly-Supervised Dual-Stream Deep Learning Framework
> for Cortical Surface Reconstruction From Diffusion MRI." *Advanced Science*
> (2026): e76596. https://doi.org/10.1002/advs.76596

If you use DDSurfer in your research, please cite:

```bibtex
@article{Li2026DDSurfer,
  author  = {Li, Chengjin and Zhang, Wei and Zhu, Xi and Chen, Yuqian and
             Sochen, Nir A. and Rushmore, Jarrett and Westin, Carl-Fredrik and
             Rathi, Yogesh and O'Donnell, Lauren J. and Pasternak, Ofer and
             Zhang, Fan},
  title   = {DDSurfer: A Weakly-Supervised Dual-Stream Deep Learning Framework
             for Cortical Surface Reconstruction From Diffusion MRI},
  journal = {Advanced Science},
  year    = {2026},
  pages   = {e76596},
  doi     = {10.1002/advs.76596},
  url     = {https://doi.org/10.1002/advs.76596}
}
```

---

## Inputs

The pipeline accepts four files, as in
[DDParcel](https://github.com/zhangfanmark/DDParcel/blob/main/process.sh):
corrected 4D DWI, bval, bvec, and a brain mask on the same native voxel grid.
No precomputed DTI or structural MRI is required.

```text
inputs/<subID>/
  dwi.nii.gz
  dwi.bval
  dwi.bvec
  mask.nii.gz
```

DDParcel-style subject filenames and the HCP `T1w/Diffusion/` layout are also
accepted. Motion/eddy/susceptibility correction and the corresponding gradient
updates must already be done.

---

## Run DDSurfer

From the repository root:

```bash
python3 run_ddsurfer_pipeline.py --subject <subID>
# Equivalent shell entry point:
bash run_ddsurfer_pipeline.sh --subject <subID>
```

For files stored elsewhere:

```bash
python3 run_ddsurfer_pipeline.py --subject <subID> \
  --dwi ./data/dwi.nii.gz --bval ./data/dwi.bval \
  --bvec ./data/dwi.bvec --mask ./data/mask.nii.gz
```

The workflow is DWI -> DTI features -> MNI inference -> native surfaces.
Volume preprocessing is provided by `preprocessing/run.sh`, including the
DTI-estimation stage in `preprocessing/dti.sh`.
Only native surfaces are retained. DTI maps, resampled volumes and MNI meshes
are intermediate cache files and are removed after success by default.
Use `--keep-cache` to retain them; failed runs retain the cache for diagnosis.
The source input files are never modified or removed.

Inputs default to `inputs/<subID>/`; results default to `outputs/<subID>/`.
Use `--raw-input-root` and `--output-root` to change these roots.
Use `--device cpu` for CPU inference or `--device cuda:0` to select a GPU.
CUDA `--precision auto` selects BF16; CPU uses FP32.

---

## Outputs

```text
outputs/<subID>/
  ddsurfer/
    lh.white.obj
    lh.pial.obj
    rh.white.obj
    rh.pial.obj
  logs/           input records, transform and execution logs
```

Surface outputs use OBJ only, in **native scanner RAS, in millimetres**.
No MNI-space surface is written outside the disposable cache.

---

## Optional Postprocessing

Postprocessing is **disabled by default**. Configure `FREESURFER_HOME` and
`FS_LICENSE`, then add `--post-process`:

```bash
python3 run_ddsurfer_pipeline.py --subject <subID> --post-process
```

`--postprocess-hemi left` or `right` selects one hemisphere;
`--postprocess-atlases aparc` selects one atlas. Otherwise both hemispheres
and the `aparc,aparc.a2009s` atlases are processed.

**DDSurfer postprocessing is plug-and-play and can be used independently.**
Its only required data inputs are four native cortical surfaces: left/right
white and pial. They may be reconstructed by DDSurfer, FreeSurfer, FastSurfer
or another method; DWI, DTI and structural MRI are not required. FreeSurfer,
its license, and `fsaverage` must be configured.

```bash
bash postprocessing/run.sh --subject <subID> \
  --lh-white ./outputs/<subID>/ddsurfer/lh.white.obj --lh-pial ./outputs/<subID>/ddsurfer/lh.pial.obj \
  --rh-white ./outputs/<subID>/ddsurfer/rh.white.obj --rh-pial ./outputs/<subID>/ddsurfer/rh.pial.obj
```

The four surfaces must share native scanner-RAS millimetre coordinates.
Each white/pial pair must preserve vertex correspondence and topology.
Indexed OBJ/PLY/OFF, STL and FreeSurfer binary surfaces are supported.
Paired STL inputs must preserve corresponding triangle order.
No nearest-neighbour correspondence is guessed.

Postprocessing adds these standard FreeSurfer directories directly under
`outputs/<subID>/`:

```text
surf/       white/pial, curvature, sulcal depth, thickness, area,
            inflated surfaces, spheres and spherical registration
label/      cortex labels, atlas annotations and regional labels
stats/      surface-only regional statistics
fsaverage/  thickness, curvature, sulcal-depth and area overlays
mri/        native reference geometry for FreeSurfer compatibility
logs/       command logs, timing and coordinate checks
```

Without an MRI, the reference under `mri/` contains geometry only, not acquired
or synthesized anatomical intensities. No segmentation-derived tissue volumes
are reported. See [postprocessing usage](postprocessing/README.md) for details.

Verify weights with `cd ckpts && sha256sum -c SHA256SUMS`.

---

## Key Dependencies

- Python 3.8+
- PyTorch and torchvision (CUDA optional for GPU acceleration)
- NumPy, SciPy, SimpleITK, nibabel, trimesh; pynrrd for DWI conversion
- Slicer with SlicerDMRI for raw-DWI processing; set `SLICER_PATH` or put `Slicer` on `PATH`.
- FreeSurfer with a valid license and `fsaverage` (optional surface postprocessing).

Python dependencies for preprocessing, inference and postprocessing are defined
in [`requirements.txt`](requirements.txt) at the repository root. The root
[`environment.yml`](environment.yml) creates a `ddsurfer` conda environment
and installs the same requirements. From the repository root:

```bash
conda env create -f environment.yml
conda activate ddsurfer
```

Alternatively, install into an existing compatible Python environment:

```bash
python -m pip install -r requirements.txt
```

Slicer/SlicerDMRI and optional FreeSurfer are external applications and must be
installed separately; they are not provided by these Python dependency files.

---

## Slicer Extension

**SlicerDDSurfer will be open-sourced soon** at
[ChengjinLii/SlicerDDSurfer](https://github.com/ChengjinLii/SlicerDDSurfer).

---

## Support

Open issues or questions can be directed through the repository’s issue tracker.

---

## Acknowledgments

This work is in part supported by the National Key R&D Program of China (No. 2023YFE0118600), the National Natural Science Foundation of China (No. 62371107).
