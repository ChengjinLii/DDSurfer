# Local Source-Build Validation

[Build guide](README.md) | [中文](VALIDATION.zh-CN.md)

## Build Verification

Validated on **2026-10-23**, Linux x86_64 / Ubuntu 22.04, GCC 11.4 and CMake
3.31.6. The real Python build command compiled all seven original FreeSurfer
command targets from pinned revision
`7eb846079b0dc0c92e8313205a3d2387b5c7a354`, with ITK Common/VNL 4.13.2 and zlib
1.3.1 built from checked source archives. The ITK compiler-detection guard was
adjusted to accept newer GCC; no numerical routine was rewritten.

The full FreeSurfer installation and the earlier precompiled-runtime experiment
were hidden using a mount namespace during compilation. No FreeSurfer activation
file was supplied to the build. All seven resulting commands passed `--version`
startup checks, and the installed manifest/file checksums were verified before
atomic publication. The build did not copy installed FreeSurfer executables.

The distributed atlas archive contains **data and license notices only**. It has
no `bin/` directory or private activation key. No precompiled native command
package is shipped in this implementation.

## Numerical Comparison

HCP **100307, left hemisphere**: 164,239 vertices and 328,474 triangles. Both
runs started independently from the same native scanner-RAS white/pial OBJ
files, using a geometry-only LIA reference, four threads, sphere seed 1234,
and the `aparc,aparc.a2009s` atlases.

- **Reference:** the existing FreeSurfer 7.4.1 installation.
- **Candidate:** the locally compiled source components, not extracted release binaries.
- **Isolation:** the reference installation, previous binary experiment and build
  directories were hidden for the candidate run. Runtime needed only the built
  commands, installed atlas data, Python dependencies and a separately provided
  activation file.

**All 139 checked file payloads matched exactly**, without a floating-point
comparison tolerance:

| Output | Result |
| --- | --- |
| White/pial, orig, smoothwm, inflated, sphere and registered sphere | Decoded vertices, faces and volume geometry identical |
| Area, curvature, sulcal depth and thickness | All scalar values identical; maximum difference 0 |
| Cortex mask | Identical vertex membership and label rows |
| `aparc` and `aparc.a2009s` | 100% vertex-label agreement; color tables and region names identical |
| Regional labels and statistics | Identical rows and numerical values |
| fsaverage overlays | Identical data, affine and numeric MGH headers |

Path/time text in surface-file headers is not required to be byte-identical.
Compiled executable hashes can also differ because upstream version reporting
contains compilation timestamps. File-byte identity, numerical identity and
anatomical accuracy are different questions. FreeSurfer output is a compatibility
reference here, **not manual anatomical ground truth**.

## Tests and Limits

The CPU suite passed **187 tests**. Coverage includes backend selection, legacy
FreeSurfer compatibility, explicit/no-network build behavior, dependency hashes,
safe archive extraction, source-edit detection, runtime integrity, private-key
exclusion, parameter forwarding and spatial geometry.

Local evidence is retained under `outputs/source-build-check/` (not tracked):
`build-command.log`, `build-command-final.log`, source/build logs under `work/`,
`native-files.trace`, `source-comparison.json`, paired subject outputs and
`tests-final.log`. Runtime manifests record the exact tested binary checksums
and toolchain information.

**Not established:** native Windows support, macOS compilation, cross-platform
numerical equivalence, a multi-dataset cohort or right-hemisphere source-build
parity. Linux validation does not imply those tests passed. Existing supported
FreeSurfer installations remain selectable instead of the local-source route.
No processing-speed improvement is claimed; original numerical algorithms and
iteration settings are retained, while initial compilation adds setup work.
