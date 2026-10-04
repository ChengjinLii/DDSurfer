<div align="center">

# DDSurfer

**Cortical surface reconstruction from diffusion MRI**

[![Python](https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white)](#key-dependencies)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](#key-dependencies)
[![NumPy](https://img.shields.io/badge/Numerics-NumPy-013243?logo=numpy&logoColor=white)](#key-dependencies)
[![SimpleITK](https://img.shields.io/badge/Imaging-SimpleITK-0098A9)](#key-dependencies)
[![SlicerDMRI](https://img.shields.io/badge/Preprocessing-SlicerDMRI-4D7EB3)](#key-dependencies)
[![FreeSurfer](https://img.shields.io/badge/Postprocessing-FreeSurfer%20%28optional%29-6A737D)](#optional-postprocessing)

[Publication](#publication) | [Overview](#overview) | [Run DDSurfer](#run-ddsurfer) | [Postprocessing](#optional-postprocessing) | [Dependencies](#key-dependencies)

</div>

DDSurfer reconstructs white and pial cortical surfaces from diffusion MRI.
Preprocessing, surface inference and optional postprocessing are provided in
one workflow, with model weights loaded automatically from `weights/`.

**Four diffusion inputs. White and pial surfaces for both hemispheres.**

---

## Publication

DDSurfer has been accepted and published online in **Advanced Science**:

> Chengjin Li, Wei Zhang, Xi Zhu, Yuqian Chen, Nir A. Sochen, Jarrett Rushmore,
> Carl-Fredrik Westin, Yogesh Rathi, Lauren J. O'Donnell, Ofer Pasternak, and
> Fan Zhang. ["DDSurfer: A Weakly-Supervised Dual-Stream Deep Learning Framework
> for Cortical Surface Reconstruction From Diffusion MRI."](https://doi.org/10.1002/advs.76596)
> *Advanced Science* (2026): e76596.

If you use DDSurfer in your research, please cite the paper above.

<details>
<summary><strong>Citation (BibTeX)</strong></summary>

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

</details>

---

## Overview

![DDSurfer framework: pseudo-ground-truth generation, dual-stream surface reconstruction, postprocessing and 3D Slicer integration.](assets/ddsurfer-overview.png)

**DDSurfer at a glance.** Weak supervision, white and pial surface reconstruction,
surface-based analysis and integration with 3D Slicer.

---

## Inputs

The pipeline takes **four files**: corrected 4D DWI, bval, bvec and a brain mask
on the same native voxel grid. No precomputed DTI or structural MRI is required.

```text
inputs/<subID>/
  dwi.nii.gz
  dwi.bval
  dwi.bvec
  mask.nii.gz
```

**Before running:** motion, eddy-current and susceptibility correction, together
with the corresponding gradient updates, must already be done.

---

## Run DDSurfer

### Quick Start

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

### Workflow

`DWI -> DTI features -> MNI inference -> native surfaces`

Preprocessing runs automatically before surface prediction.

Default paths:

- Inputs: `inputs/<subID>/`
- Outputs: `outputs/<subID>/`

### Common Options

| Option | Usage |
| --- | --- |
| `--raw-input-root <path>` | Change the input directory. |
| `--output-root <path>` | Change the output directory. |
| `--device cuda:0` or `--device cpu` | Select GPU or CPU inference. |
| `--precision auto` | Use BF16 for CUDA inference (requires BF16 support), or FP32 on CPU. |
| `--preprocess-jobs 2` | Run independent DTI stages concurrently; default: 1. |
| `--keep-cache` | Retain intermediate files and detailed stage logs. |

DTI maps and surfaces are saved automatically. Intermediate cache is removed
after success; failed runs retain it for diagnosis. Source inputs are never
modified or removed.

To verify the bundled weights:

```bash
(cd weights && sha256sum -c SHA256SUMS)
```

---

## Outputs

```text
outputs/<subID>/
  dti/
    <subID>-FA.nii.gz
    <subID>-MD.nii.gz
    <subID>-MinEigenvalue.nii.gz
    <subID>-MidEigenvalue.nii.gz
    <subID>-MaxEigenvalue.nii.gz
    <subID>-Trace.nii.gz
  ddsurfer/
    lh.white.obj
    lh.pial.obj
    rh.white.obj
    rh.pial.obj
  logs/           run summary, input records and execution log
```

---

## Optional Postprocessing

### Run with DDSurfer

Postprocessing is **disabled by default**. Configure `FREESURFER_HOME` and
`FS_LICENSE`, then add `--post-process`:

```bash
python3 run_ddsurfer_pipeline.py --subject <subID> --post-process
```

`--postprocess-hemi left` or `right` selects one hemisphere;
`--postprocess-atlases aparc` selects one atlas. Otherwise both hemispheres
and the `aparc,aparc.a2009s` atlases are processed.

### Use Your Own Surfaces

**DDSurfer postprocessing is plug-and-play and can be used independently.**
Its only required data inputs are four native cortical surfaces: left/right
white and pial. They may be reconstructed by DDSurfer, FreeSurfer, FastSurfer
or another method; DWI, DTI and structural MRI are not required. FreeSurfer,
its license, and `fsaverage` must be configured.

```bash
bash postprocessing/run.sh --subject <subID> \
  --lh-white ./outputs/<subID>/ddsurfer/lh.white.obj \
  --lh-pial ./outputs/<subID>/ddsurfer/lh.pial.obj \
  --rh-white ./outputs/<subID>/ddsurfer/rh.white.obj \
  --rh-pial ./outputs/<subID>/ddsurfer/rh.pial.obj
```

The four surfaces must share native scanner-RAS millimetre coordinates.
Each white/pial pair must preserve vertex correspondence and topology.
Indexed OBJ/PLY/OFF, STL and FreeSurfer binary surfaces are supported.
Paired STL inputs must preserve corresponding triangle order.
No nearest-neighbour correspondence is guessed.

### Postprocessing Results

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

---

## Key Dependencies

| Component | Purpose | Requirement |
| --- | --- | --- |
| Python 3.8+ | Pipeline and utilities | Required |
| PyTorch, torchvision | Surface inference | Required; CUDA optional |
| NumPy, SciPy, SimpleITK, nibabel, trimesh, pynrrd | Image and surface processing | Required |
| Slicer with SlicerDMRI | Raw-DWI processing | Set `SLICER_PATH` or put `Slicer` on `PATH` |
| FreeSurfer | Surface postprocessing | Optional; license and `fsaverage` required when enabled |

### Python Environment

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

For questions, bug reports or feature requests, please open a
[GitHub issue](https://github.com/ChengjinLii/DDSurfer/issues).

---

## Acknowledgments

This work is supported in part by the National Key R&D Program of China
(No. 2023YFE0118600) and the National Natural Science Foundation of China
(No. 62371107).
