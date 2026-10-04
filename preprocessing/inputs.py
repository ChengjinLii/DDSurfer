"""Resolve and validate the DWI, gradient table and native brain mask."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np

from utils.files import sha256_file


INPUT_NAMES = ('dwi', 'bval', 'bvec', 'mask')


def resolve_inputs(subject, input_root=None, *, allow_missing_mask=False, **files):
    if Path(subject).name != subject or subject in ('', '.', '..'):
        raise ValueError('Invalid subject identifier')
    supplied = [files.get(name) is not None for name in INPUT_NAMES]
    if any(supplied):
        if not all(supplied) and not (allow_missing_mask and all(supplied[:3])):
            raise ValueError('Provide --dwi, --bval, --bvec and --mask together')
        result = {name: Path(files[name]).expanduser().resolve()
                  for name in INPUT_NAMES if files.get(name) is not None}
    else:
        if input_root is None:
            raise ValueError('Provide all four input files or an input root')
        root = Path(input_root).expanduser().resolve()
        result = None
        maskless = None
        for directory in (root / subject / 'T1w/Diffusion', root / subject, root):
            for extension in ('.nii.gz', '.nii'):
                for mask_extension in ('.nii.gz', '.nii'):
                    for names in ((f'dwi{extension}', 'dwi.bval', 'dwi.bvec', f'mask{mask_extension}'),
                                  (f'dwi{extension}', 'bval', 'bvec', f'mask{mask_extension}'),
                                  (f'{subject}{extension}', f'{subject}.bval', f'{subject}.bvec', f'{subject}-mask{mask_extension}'),
                                  (f'data{extension}', 'bvals', 'bvecs', f'nodif_brain_mask{mask_extension}')):
                        candidate = dict(zip(INPUT_NAMES, (directory / name for name in names)))
                        if all(path.is_file() for path in candidate.values()):
                            result = candidate
                            break
                        if allow_missing_mask and maskless is None and all(
                                candidate[name].is_file() for name in INPUT_NAMES[:3]):
                            maskless = {name: candidate[name] for name in INPUT_NAMES[:3]}
                    if result is not None:
                        break
                if result is not None:
                    break
            if result is not None:
                break
        if result is None and allow_missing_mask:
            result = maskless
        if result is None:
            raise FileNotFoundError(f'No DWI/bval/bvec/mask bundle found under {root}; '
                                    'provide the file paths explicitly, or use --auto-mask if only the mask is missing')
    result = {name: path.resolve() for name, path in result.items()}
    for name, path in result.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'{name} input is missing or empty: {path}')
        if '\n' in str(path) or '\r' in str(path):
            raise ValueError('Input paths must not contain line breaks')
    return result


def _validate_geometry(name, image):
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError(f'Invalid {name} affine')
    q, qc = image.get_qform(coded=True)
    s, sc = image.get_sform(coded=True)
    if qc and sc and not np.allclose(q, s, atol=1e-4, rtol=0):
        raise ValueError(f'{name} qform and sform disagree')
    if image.header.get_xyzt_units()[0] not in ('unknown', 'mm'):
        raise ValueError(f'{name} spatial units must be millimetres')


def validate_diffusion_inputs(files):
    if not str(files['dwi']).endswith(('.nii', '.nii.gz')):
        raise ValueError('dwi must be a NIfTI file (.nii or .nii.gz)')
    dwi = nib.load(str(files['dwi']))
    if len(dwi.shape) != 4 or dwi.shape[3] < 7:
        raise ValueError('DWI must be a 4D NIfTI with diffusion directions and b0 volumes, not a DTI/scalar map')
    _validate_geometry('dwi', dwi)
    bvals = np.atleast_1d(np.loadtxt(str(files['bval']))).reshape(-1)
    bvecs = np.atleast_2d(np.loadtxt(str(files['bvec'])))
    count = dwi.shape[3]
    if bvecs.shape == (3, count):
        bvecs = bvecs.T
    if bvals.shape != (count,) or bvecs.shape != (count, 3):
        raise ValueError('DWI volume count must match bval entries and bvec directions (3xN or Nx3)')
    if not np.isfinite(bvals).all() or np.any(bvals < 0) or not np.isfinite(bvecs).all():
        raise ValueError('Gradient table must be finite with nonnegative b-values')
    weighted = bvals > 50.
    if weighted.all() or weighted.sum() < 6:
        raise ValueError('Tensor estimation needs a b0 volume (b <= 50) and at least six diffusion directions')
    norms = np.linalg.norm(bvecs[weighted], axis=1)
    if np.any(norms < 1e-6):
        raise ValueError('Diffusion-weighted volumes must have nonzero bvec directions')
    directions = bvecs[weighted] / norms[:, None]
    x, y, z = directions.T
    design = np.column_stack([x*x, y*y, z*z, 2*x*y, 2*x*z, 2*y*z])
    if np.linalg.matrix_rank(design) < 6:
        raise ValueError('Gradient directions do not span the six tensor coefficients')
    return dict(shape=list(dwi.shape), affine=dwi.affine.tolist(),
                b0_volumes=int((~weighted).sum()), diffusion_volumes=int(weighted.sum()))


def validate_mask(dwi_path, mask_path):
    if not str(mask_path).endswith(('.nii', '.nii.gz')):
        raise ValueError('mask must be a NIfTI file (.nii or .nii.gz)')
    dwi, mask = nib.load(str(dwi_path)), nib.load(str(mask_path))
    if len(mask.shape) != 3 or mask.shape != dwi.shape[:3]:
        raise ValueError('The 3D brain mask must have the same voxel grid as the DWI')
    _validate_geometry('mask', mask)
    if not np.allclose(dwi.affine, mask.affine, atol=1e-4, rtol=0):
        raise ValueError('Brain mask and DWI are not in the same physical space; resample the mask to the DWI grid first')
    data = np.asarray(mask.dataobj)
    if not np.isfinite(data).all() or np.any(data < 0) or not np.any(data > 0):
        raise ValueError('Brain mask must be finite, nonnegative and nonempty')


def validate_inputs(files):
    geometry = validate_diffusion_inputs(files)
    validate_mask(files['dwi'], files['mask'])
    return geometry


def fingerprint(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256_file(path))


def ensure_record(path, signature):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise ValueError(f'Cached outputs have different raw inputs/settings; use a new output directory: {path.parent}')
        return
    if path.parent.exists() and any(path.parent.iterdir()):
        raise ValueError(f'Existing outputs have no raw-input record; use a new output directory: {path.parent}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(signature, indent=2) + '\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject')
    parser.add_argument('--input-root', type=Path)
    for name in INPUT_NAMES:
        parser.add_argument(f'--{name}', type=Path)
    parser.add_argument('--reference-image', type=Path)
    parser.add_argument('--mask-flip', type=int, default=1)
    parser.add_argument('--record', type=Path)
    parser.add_argument('--reuse-record', type=Path)
    parser.add_argument('--print-paths', action='store_true')
    args = parser.parse_args(argv)
    if args.reuse_record is not None:
        if args.record is None:
            parser.error('--reuse-record requires --record')
        ensure_record(args.record, json.loads(args.reuse_record.read_text()))
        return
    if not args.subject or (args.input_root is None and args.dwi is None):
        parser.error('Specify --subject and an input root or all four input files')
    files = resolve_inputs(args.subject, args.input_root, **{name: getattr(args, name) for name in INPUT_NAMES})
    geometry = validate_inputs(files)
    if args.record is not None:
        if args.reference_image is None:
            parser.error('--record requires --reference-image')
        signature = dict(inputs={name: fingerprint(path) for name, path in files.items()},
                         reference=fingerprint(args.reference_image), tensor_estimation='WLS',
                         mask_flip=args.mask_flip, geometry=geometry)
        ensure_record(args.record, signature)
    if args.print_paths:
        for name in INPUT_NAMES:
            print(files[name])


if __name__ == '__main__':
    main()
