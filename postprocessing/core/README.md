# Local Surface Components

**Compile the original surface algorithms on the user's machine, or use an existing FreeSurfer installation.**

[Postprocessing](../README.md) | [Validation](VALIDATION.md) | [中文](README.zh-CN.md)

No precompiled Linux executable package is shipped. Python drives seven original
C++ command targets; geometry preparation, regional label export, surface-only
statistics, scheduling and reuse remain in Python. Atlas data are supplied
separately in [`../assets/`](../assets/README.md).

---

## Choose a Backend

| Backend | Behavior |
| --- | --- |
| `auto` | Use a configured FreeSurfer installation; otherwise use completed local components. |
| `native` | Require locally compiled components; ignore `FREESURFER_HOME` for postprocessing. |
| `freesurfer` | Require an existing installation selected by `--freesurfer-home` or `FREESURFER_HOME`. |

An explicit `--runtime` selects a custom local installation. The main entry uses
`--postprocess-runtime` and `--postprocess-backend`. Postprocessing is still off
by default. Normal inference and environment checks **never download or compile
source code**; missing components produce a setup instruction before processing.

## Build Once

Install a C/C++ compiler, Git and CMake >=3.24,<4. For Ubuntu/Debian:

```bash
sudo apt-get install build-essential git libcrypt-dev
python -m pip install 'cmake>=3.24,<4'
python -m postprocessing.core.build --jobs 4
```

The explicit build command downloads pinned FreeSurfer, ITK and zlib sources,
checks source revisions/archive SHA256 hashes, builds the required native targets
and validates the installed files. It does not install the full FreeSurfer suite,
GUI, Python distribution or reconstruction pipeline. Only the ITK Common/VNL
base needed by this build is enabled, not the full ITK module collection.
The pinned ITK compiler guard is adjusted for newer GCC; numerical routines and
FreeSurfer algorithm sources are not rewritten.

The build is CPU-only. `--jobs` bounds compilation concurrency; four is the
default maximum. The first build needs network access, disk space for source,
intermediate objects and dependencies, and may take several minutes or longer.
It does not modify a running conda environment or install Python packages.

Completed components and build files are kept under
`${XDG_CACHE_HOME:-~/.cache}/ddsurfer/postprocessing/`. Use `--work-dir` and
`--prefix` for custom, separate build/install locations, or `--cmake` to select
an executable. `--source-root` accepts a clean checkout of the pinned revision;
`--offline` requires that checkout and the checked dependency archives already
be cached. Failed builds retain logs and intermediates for diagnosis/resume.
A runtime is published only after all seven commands build and pass startup and
integrity checks. Existing installations are not overwritten.

The source path is being validated on Linux. **macOS is not yet validated;
native Windows is unsupported** because the upstream code uses POSIX facilities.
WSL2 is a separate Linux execution route, not native Windows compatibility.
Source distribution alone does not establish cross-platform numerical parity.

## Run

Provide your own valid FreeSurfer activation file outside the source/runtime:

```bash
export FS_LICENSE=./private/freesurfer-license.txt

python run_ddsurfer_pipeline.py --subject <subID> --post-process \
  --postprocess-backend native
```

Or process four existing native scanner-RAS surfaces independently:

```bash
bash postprocessing/run.sh --subject <subID> --native-root ./outputs \
  --output-root ./postprocessed --backend native --threads 4
```

To use an installed FreeSurfer instead:

```bash
python run_ddsurfer_pipeline.py --subject <subID> --post-process \
  --postprocess-backend freesurfer --freesurfer-home ./software/freesurfer
```

For that backend, its own `license.txt` is accepted if `FS_LICENSE` is unset.
Optional SynthStrip masking remains a separate configuration; locally compiling
these surface tools does not provide SynthStrip or its model.

## Retained Commands

| Native command | Purpose |
| --- | --- |
| `mris_curvature` | Mean/Gaussian curvature |
| `mris_place_surface` | Area, smoothed curvature and thickness |
| `mris_inflate` | Inflation and sulcal depth |
| `mris_sphere` | Sphere generation |
| `mris_register` | Folding-feature atlas registration |
| `mri_label2label` | Cortex-mask transfer between surfaces |
| `mri_surf2surf` | Annotation and scalar transfer |

`mri_annotation2label`, `mris_anatomical_stats`, `mri_convert`, `mris_convert` and
`recon-all` are not installed by this builder. The Python statistics are the
documented **surface-only tables**, not a replacement for all recon-all metrics.
Cortex-mask transfer differs from exporting regions from an existing annotation;
the former still uses `mri_label2label`. Python overlay packing does not replace
the spherical correspondence in `mri_surf2surf`.

## Reproducibility and Licensing

Use the same surface/reference inputs, thread settings and `--sphere-seed 1234`
(main entry: `--postprocess-sphere-seed 1234`) for controlled comparisons. The
existing unset-seed default is preserved. Recompiling on a different toolchain
or platform does not guarantee bitwise-identical floating-point output.

```bash
python -m postprocessing.core.runtime --runtime ./local-components
python -m postprocessing.core.compare \
  --reference ./reference/<subID> --candidate ./postprocessed/<subID> \
  --hemi left --output ./outputs/surface-comparison.json
```

The source revision, dependency versions, build recipe and installed checksums
are recorded locally. Runtime integrity is checked before use. No private
activation key is copied or bypassed. FreeSurfer-derived code/data and third-party
libraries retain their licenses; they are not relicensed under DDSurfer's MIT
license. The native algorithms are reused, not a clean-room reimplementation.

- [Pinned upstream source](https://github.com/freesurfer/freesurfer/tree/7eb846079b0dc0c92e8313205a3d2387b5c7a354)
- [Upstream build guide](https://surfer.nmr.mgh.harvard.edu/fswiki/BuildGuide)
- [FreeSurfer license](../assets/LICENSE.txt)
- [FreeSurfer activation registration](https://surfer.nmr.mgh.harvard.edu/registration.html)
