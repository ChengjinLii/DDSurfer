"""Mask, resample and normalize scalar channels with reusable stage outputs."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib

from preprocessing.cache import validate_output
from preprocessing.mask import skull_strip
from preprocessing.normalize import normalize_volume
from preprocessing.resample import resample_image_to_target_space
from utils.stages import StageCache, subject_lock

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = (('MinEigenvalue', 'MinEigenvalue'), ('MidEigenvalue', 'MidEigenvalue'),
            ('FractionalAnisotropy', 'FA'), ('MaxEigenvalue', 'MaxEigenvalue'), ('MeanDiffusivity', 'MD'))
GEOMETRY = dict(target_spacing=(1., 1., 1.), target_size=(176, 224, 176),
                target_origin=(88.29999542236328, 129., -69.),
                target_direction=(-1., 0., 0., 0., -1., 0., 0., 0., 1.))


def run(args):
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    intermediate = folder / '.preprocessing'
    cache = StageCache(intermediate / 'stages.json', validate=validate_output)
    code = [Path(__file__), ROOT / 'preprocessing/cache.py', ROOT / 'utils/stages.py']
    stem = args.subject
    source_mask = args.input_dir / f'{stem}-mask-Reg.nii.gz'
    target_mask = folder / f'{stem}-brainmask_resampled.nii.gz'
    source_mask_data = nib.load(str(source_mask)).get_fdata().astype(bool)
    default_helpers = {kind: ROOT / 'preprocessing' / name for kind, name in
                       (('mask', 'mask.py'), ('resample', 'resample.py'), ('normalize', 'normalize.py'))}

    def process(kind, inputs, output, action, cli):
        helper = getattr(args, kind + '_script')
        if helper.resolve() != default_helpers[kind].resolve():
            action = lambda staged: subprocess.run([sys.executable, str(helper)] +
                [str(staged[0]) if part == str(output) else part for part in cli], check=True)
        cache.run(f'{kind}.{output.name}', list(inputs) + code + [helper], [output], action,
                  settings=GEOMETRY if kind == 'resample' else dict(epsilon=1e-6, zero_background=True))

    def resample(source, output):
        cli = ['--source_image_path', str(source), '--output_file_path', str(output)]
        for name, values in GEOMETRY.items():
            cli += ['--' + name] + list(map(str, values))
        process('resample', [source], output,
                lambda staged: resample_image_to_target_space(source, staged[0], **GEOMETRY), cli)

    resample(source_mask, target_mask)
    target_mask_data = nib.load(str(target_mask)).get_fdata().astype(bool)
    channels = CHANNELS + (() if args.minimal else (('Trace', 'Trace'),))
    transform = args.input_dir / f'{stem}-b0ToAtlasT2.tfm'
    cache.run('transform', [transform], [folder / transform.name],
              lambda staged: shutil.copyfile(transform, staged[0]))
    for scalar, suffix in channels:
        source = args.input_dir / f'{stem}-dti-{scalar}-Reg.nii.gz'
        masked = args.input_dir / f'{stem}-dti-{scalar}-Reg-masked.nii.gz'
        process('mask', [source, source_mask], masked,
                lambda staged: skull_strip(source, source_mask, staged[0], mask_data=source_mask_data),
                ['--input_path', str(source), '--mask_path', str(source_mask), '--output_path', str(masked)])
        resampled = intermediate / f'{stem}-{suffix}.nii.gz'
        resample(masked, resampled)
        output = folder / f'{stem}-{suffix}.nii.gz'
        process('normalize', [resampled, target_mask], output,
                lambda staged: normalize_volume(resampled, staged[0], mask=target_mask_data),
                ['--input_file', str(resampled), '--mask_file', str(target_mask), '--output_file', str(output)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--minimal', action='store_true')
    for kind, name in (('mask', 'mask.py'), ('resample', 'resample.py'), ('normalize', 'normalize.py')):
        parser.add_argument('--' + kind + '-script', type=Path, default=ROOT / 'preprocessing' / name)
    args = parser.parse_args(argv)
    with subject_lock(args.output_dir / '.preprocessing/lock'):
        run(args)


if __name__ == '__main__':
    main()
