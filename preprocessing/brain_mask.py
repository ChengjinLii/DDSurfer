"""Optionally extract a native DWI brain mask with FreeSurfer SynthStrip."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np

from preprocessing.inputs import fingerprint, resolve_inputs, validate_diffusion_inputs, validate_mask
from utils.files import atomic_json, sha256_file
from utils.stages import subject_lock

B0_THRESHOLD = 50.


def save_native_image(data, source, destination):
    image = source.__class__(data, source.affine, source.header.copy())
    image.set_data_dtype(data.dtype)
    image.header.set_slope_inter(1., 0.)
    image.set_qform(source.get_qform(), int(source.header['qform_code']))
    image.set_sform(source.get_sform(), int(source.header['sform_code']))
    nib.save(image, str(destination))


def mean_b0(files, destination):
    validate_diffusion_inputs(files)
    if Path(destination).is_symlink() or Path(destination).resolve() in {
            Path(path).resolve() for path in files.values()}:
        raise ValueError('The mean b0 must not overwrite a source input or a symlink')
    image = nib.load(str(files['dwi']), keep_file_open=True)
    indices = np.flatnonzero(np.loadtxt(str(files['bval'])).reshape(-1) <= B0_THRESHOLD)
    mean = np.zeros(image.shape[:3], np.float64)
    # Read b0 volumes in acquisition order, without loading the entire DWI.
    for index in indices:
        volume = np.asarray(image.dataobj[..., int(index)], dtype=np.float32)
        if not np.isfinite(volume).all():
            raise ValueError('The b0 volumes contain nonfinite values')
        mean += volume
    mean = (mean / len(indices)).astype(np.float32)
    if not np.any(mean > 0):
        raise ValueError('The mean b0 has no positive signal')
    save_native_image(mean, image, destination)
    return len(indices)


def synthstrip_environment(freesurfer_home, threads):
    env = dict(os.environ)
    if freesurfer_home is not None:
        home = Path(freesurfer_home).expanduser().resolve()
        env['FREESURFER_HOME'] = str(home)
        env['PATH'] = str(home / 'bin') + os.pathsep + env.get('PATH', '')
        executable = home / 'bin/mri_synthstrip'
    else:
        found = shutil.which('mri_synthstrip')
        if found is None:
            raise FileNotFoundError('Automatic masking requires FreeSurfer mri_synthstrip; '
                                    'set FREESURFER_HOME or pass --freesurfer-home')
        executable = Path(found)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f'SynthStrip is missing or not executable: {executable}')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[name] = str(threads)
    return executable, env


def generate_mask(files, output, work_dir, record, *, freesurfer_home=None, threads=4):
    if threads < 1:
        raise ValueError('--threads must be positive')
    validate_diffusion_inputs(files)
    output, work_dir, record = map(Path, (output, work_dir, record))
    if not str(output).endswith(('.nii', '.nii.gz')):
        raise ValueError('The generated mask must be a NIfTI file')
    if any(path.is_symlink() for path in (output, output.parent, work_dir, record, record.parent)):
        raise ValueError('Refusing to write through a symlinked mask or cache')
    if output.resolve() in {Path(path).resolve() for path in files.values()}:
        raise ValueError('The generated mask must not overwrite a source input')
    if record.resolve() == output.resolve() or record.resolve() in {
            Path(path).resolve() for path in files.values()}:
        raise ValueError('The mask record must be separate from image and source files')
    record.parent.mkdir(parents=True, exist_ok=True)
    with subject_lock(record.with_suffix('.lock')):
        signature = dict(method='freesurfer_synthstrip', b0_threshold=B0_THRESHOLD,
                         inputs={name: fingerprint(files[name]) for name in ('dwi', 'bval', 'bvec')})
        if record.exists():
            saved = json.loads(record.read_text())
            if saved['signature'] != signature:
                raise ValueError('Generated mask has different inputs; use a new output directory')
            if output.exists():
                if fingerprint(output) != saved['output']:
                    raise ValueError('Generated mask has changed; use a new output directory')
                validate_mask(files['dwi'], output)
                print(f'Reusing native brain mask: {output}', flush=True)
                return output
        elif output.exists():
            raise ValueError('Existing mask has no generation record; supply it with --mask '
                             'or use a new output directory')

        executable, env = synthstrip_environment(freesurfer_home, threads)
        tool = dict(executable=fingerprint(executable))
        home = env.get('FREESURFER_HOME')
        if home and (Path(home) / 'models/synthstrip.1.pt').is_file():
            tool['model_sha256'] = sha256_file(Path(home) / 'models/synthstrip.1.pt')
        work_dir.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        baseline = work_dir / 'mean_b0.nii.gz'
        count = mean_b0(files, baseline)
        # Omit -g: mask generation runs on CPU independently of inference.
        with tempfile.TemporaryDirectory(prefix='.brain-mask-', dir=output.parent) as tmp:
            raw = Path(tmp) / 'synthstrip.nii.gz'
            subprocess.run([str(executable), '-i', str(baseline), '-m', str(raw)],
                           env=env, check=True)
            validate_mask(files['dwi'], raw)
            data = np.asarray(nib.load(str(raw)).dataobj)
            if not np.all((data == 0) | (data == 1)):
                raise ValueError('SynthStrip did not produce a binary mask')
            staged = Path(tmp) / output.name
            save_native_image(data.astype(np.uint8), nib.load(str(files['dwi'])), staged)
            validate_mask(files['dwi'], staged)
            os.replace(staged, output)
        atomic_json(record, dict(signature=signature, output=fingerprint(output),
                                 b0_volumes=count, tool=tool))
        print(f'Saved native brain mask: {output}', flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dwi', 'bval', 'bvec'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--freesurfer-home', type=Path, default=os.environ.get('FREESURFER_HOME'))
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    files = resolve_inputs('brain-mask', allow_missing_mask=True,
                           **{name: getattr(args, name) for name in ('dwi', 'bval', 'bvec')})
    generate_mask(files, args.output, args.work_dir, args.record,
                  freesurfer_home=args.freesurfer_home, threads=args.threads)


if __name__ == '__main__':
    main()
