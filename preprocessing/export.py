"""Export unnormalized DTI scalar maps on the original DWI voxel grid."""
import argparse
import json
import logging
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from preprocessing.cache import related_files, validate_output
from utils.files import atomic_json, sha256_file
from utils.stages import StageCache, subject_lock

SCALARS = (
    ('FractionalAnisotropy', 'FA'),
    ('MeanDiffusivity', 'MD'),
    ('MinEigenvalue', 'MinEigenvalue'),
    ('MidEigenvalue', 'MidEigenvalue'),
    ('MaxEigenvalue', 'MaxEigenvalue'),
    ('Trace', 'Trace'),
)


def check_geometry(image, geometry):
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Native DTI maps must be scalar 3D images')
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    affine[:3, 3] = image.GetOrigin()
    affine = np.diag([-1., -1., 1., 1.]) @ affine
    if tuple(image.GetSize()) != tuple(geometry['shape'][:3]) or not np.allclose(
            affine, geometry['affine'], atol=1e-4, rtol=0):
        raise ValueError('DTI map does not match the original DWI voxel grid and physical space')


def export_native(subject, input_dir, output_dir, record=None):
    if Path(subject).name != subject or subject in ('', '.', '..'):
        raise ValueError('Invalid subject identifier')
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).absolute()
    if output_dir.is_symlink():
        raise ValueError('Refusing to export DTI through a symlinked directory')
    record = Path(record) if record is not None else input_dir / 'raw_inputs.json'
    inputs = json.loads(record.read_text())
    geometry = inputs['geometry']
    sources = {}
    for scalar, name in SCALARS:
        source = next((input_dir / f'{subject}-dti-{scalar}{suffix}'
                       for suffix in ('.nhdr', '.nrrd', '.nii.gz', '.nii')
                       if (input_dir / f'{subject}-dti-{scalar}{suffix}').is_file()), None)
        if source is None:
            if name == 'Trace':
                logging.warning('No native Trace map in this older cache; exporting the five existing scalar maps.')
                continue
            raise FileNotFoundError(f'Missing native DTI scalar: {subject}-dti-{scalar}')
        sources[name] = source
    logs = output_dir.parent / 'logs'
    cache_root = output_dir.parent / '.cache'
    if cache_root.is_symlink():
        raise ValueError('Refusing to write through a symlinked cache directory')
    state = cache_root / 'state/native_dti'
    implementation = [Path(__file__), Path(__file__).with_name('cache.py'),
                      Path(__file__).resolve().parents[1] / 'utils/stages.py']
    with subject_lock(state / '.lock'):
        cache = StageCache(state / 'stages.json', related_files, validate_output)
        maps = {}
        for name, source in sources.items():
            output = output_dir / f'{subject}-{name}.nii.gz'
            if output.is_symlink():
                raise ValueError('Refusing to replace a symlinked DTI output')

            def convert(staged, source=source):
                image = sitk.ReadImage(str(source))
                check_geometry(image, geometry)
                if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
                    raise ValueError(f'Nonfinite native scalar map: {source}')
                # Only change the file format: no masking, interpolation or normalization.
                sitk.WriteImage(image, str(staged[0]), True)
                check_geometry(sitk.ReadImage(str(staged[0])), geometry)
                if not np.array_equal(sitk.GetArrayViewFromImage(image).transpose(2, 1, 0),
                                      np.asarray(nib.load(str(staged[0])).dataobj)):
                    raise ValueError('NIfTI export changed native DTI scalar values')

            cache.run(f'native.{name}', [source, record] + implementation, [output], convert,
                      settings=dict(coordinate_space='native_scanner_RAS_mm', normalized=False))
            maps[name] = dict(file=f'dti/{output.name}', sha256=sha256_file(output))
        metadata = dict(subject=subject, coordinate_space='native_scanner_RAS_mm',
                        normalized=False, geometry=geometry, scalar_maps=maps,
                        input_dwi_sha256=inputs['inputs']['dwi']['sha256'])
        atomic_json(logs / 'dti.json', metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--record', type=Path, help='Original DWI input record; defaults to the DTI cache record.')
    args = parser.parse_args(argv)
    export_native(args.subject, args.input_dir, args.output_dir, args.record)


if __name__ == '__main__':
    main()
