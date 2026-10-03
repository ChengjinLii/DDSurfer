"""Execute DTI stages with unchanged Slicer commands and per-output receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.cache import related_files, validate_output
from utils.stages import StageCache, subject_lock

ROOT = Path(__file__).resolve().parents[1]
SCALARS = ('FractionalAnisotropy', 'MinEigenvalue', 'MidEigenvalue', 'MaxEigenvalue', 'MeanDiffusivity')


def run(args):
    folder = args.output_dir.resolve()
    stem = args.subject
    cache = StageCache(folder / '.stages.json', related_files, validate_output)
    slicer = [str(args.slicer_path / 'Slicer'), '--launch']
    helper_env = dict(os.environ)
    for name, value in (('OMP_NUM_THREADS', '1'), ('OPENBLAS_NUM_THREADS', '1'),
                        ('MKL_NUM_THREADS', '1'), ('NUMEXPR_NUM_THREADS', '1'),
                        ('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', '1'),
                        ('KMP_AFFINITY', 'disabled'), ('KMP_BLOCKTIME', '0'), ('OMP_WAIT_POLICY', 'PASSIVE')):
        helper_env.setdefault(name, value)

    def execute(key, command, inputs, outputs, helper=False, link=None):
        def action(staged):
            replacements = dict(zip(map(str, outputs), map(str, staged)))
            if link is not None:
                (staged[0].parent / link.name).symlink_to(link.resolve())
            actual = [replacements.get(str(part), str(part)) for part in command]
            subprocess.run(actual, check=True, env=helper_env if helper else None)
        implementation = [Path(__file__), ROOT / 'preprocessing/cache.py', ROOT / 'utils/stages.py']
        implementation += [Path(command[1])] if helper else [Path(command[0]), Path(command[2])]
        cache.run(key, list(inputs) + implementation, outputs, action, settings=list(map(str, command)))

    dwi_header = folder / f'{stem}.nhdr'
    mask_header = folder / f'{stem}-mask.nhdr'
    converter = ROOT / 'preprocessing/conversion/nhdr_write.py'
    execute('dwi.header', [sys.executable, converter, '--nifti', args.dwi, '--bval', args.bval,
                          '--bvec', args.bvec, '--nhdr', dwi_header],
            [args.dwi, args.bval, args.bvec, ROOT / 'preprocessing/conversion/bval_bvec_io.py'],
            [dwi_header], helper=True, link=args.dwi)
    execute('mask.header', [sys.executable, converter, '--nifti', args.mask, '--nhdr', mask_header],
            [args.mask], [mask_header], helper=True, link=args.mask)
    tensor, b0 = (folder / f'{stem}-{kind}.nhdr' for kind in ('dti', 'b0'))
    execute('tensor', slicer + [args.dmri_cli / 'DWIToDTIEstimation', '--enumeration', 'WLS',
                               dwi_header, tensor, b0], [dwi_header], [tensor, b0])
    scalars = SCALARS + (() if args.minimal else ('Trace',))

    def scalar(kind):
        output = folder / f'{stem}-dti-{kind}.nhdr'
        execute(f'scalar.{kind}', slicer + [args.dmri_cli / 'DiffusionTensorScalarMeasurements',
                                           '--enumeration', kind, tensor, output], [tensor], [output])

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(scalar, scalars))
    transform = folder / f'{stem}-b0ToAtlasT2.tfm'
    execute('registration', slicer + [args.core_cli / 'BRAINSFit', '--fixedVolume', args.reference_image,
                                     '--movingVolume', b0, '--linearTransform', transform, '--useRigid', '--useAffine'],
            [b0, args.reference_image], [transform])

    def registered(kind):
        source = args.mask if kind == 'mask' else folder / f'{stem}-dti-{kind}.nhdr'
        output = folder / (f'{stem}-mask-Reg.nii.gz' if kind == 'mask' else f'{stem}-dti-{kind}-Reg.nii.gz')
        execute(f'resample.{kind}', slicer + [args.core_cli / 'ResampleScalarVectorDWIVolume',
                '-i', 'nn' if kind == 'mask' else 'linear', source, '--Reference', args.reference_image,
                '--transformationFile', transform, output], [source, args.reference_image, transform], [output])

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(registered, scalars + ('mask',)))
    if not args.minimal:
        for kind in ('FractionalAnisotropy', 'Trace', 'MinEigenvalue', 'MidEigenvalue'):
            source = folder / f'{stem}-dti-{kind}-Reg.nii.gz'
            mask = folder / f'{stem}-mask-Reg.nii.gz'
            output = folder / f'{stem}-dti-{kind}-Reg-NormMasked.nii.gz'
            execute(f'normalize.{kind}', [sys.executable, ROOT / 'preprocessing/normalize_dti.py',
                    '--input', source, '--mask', mask, '--output', output, '--flip', str(args.mask_flip)],
                    [source, mask], [output], helper=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', required=True)
    for name in ('output-dir', 'dwi', 'bval', 'bvec', 'mask', 'slicer-path', 'dmri-cli', 'core-cli', 'reference-image'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mask-flip', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--minimal', action='store_true')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    with subject_lock(args.output_dir / '.dti.lock'):
        run(args)


if __name__ == '__main__':
    main()
