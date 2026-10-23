"""Check installation prerequisites without processing data or loading models."""

import argparse
from collections import Counter
import importlib
import os
from pathlib import Path
import re
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULES = {'pynrrd': 'nrrd'}
SLICER_INSTALL = (
    'Install 3D Slicer: https://download.slicer.org/',
    'export SLICER_PATH="/path/to/Slicer-installation"',
    'If DTI_SLICER_PATH is set, it takes precedence; update it or run: unset DTI_SLICER_PATH',
)
DMRI_INSTALL = (
    'Install SlicerDMRI through Slicer Extensions Manager, then restart Slicer.',
    'Instructions: https://dmri.slicer.org/download/',
)
FS_INSTALL = (
    'Install FreeSurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall',
    'export FREESURFER_HOME="/path/to/freesurfer"',
    'source "$FREESURFER_HOME/SetUpFreeSurfer.sh"',
)
FS_LICENSE = (
    'Request a license: https://surfer.nmr.mgh.harvard.edu/registration.html',
    'export FS_LICENSE="/path/to/license.txt"',
)
POSTPROCESS_INSTALL = (
    'Compile the native components locally: python -m postprocessing.core.build --jobs 4',
    'Or select an existing installation with --postprocess-backend freesurfer --freesurfer-home /path/to/freesurfer',
    'Set FS_LICENSE="/path/to/license.txt"; see postprocessing/README.md.',
)


class Report:
    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self.counts = Counter()
        self.hints = []

    def add(self, status, name, detail, hints=()):
        self.counts[status] += 1
        detail = str(detail).replace('\n', ' ')
        print(f'[{status}] {name}: {detail}', file=self.stream)
        for hint in hints:
            if hint not in self.hints:
                self.hints.append(hint)

    def finish(self):
        if self.hints:
            print('\nSuggested fixes (not executed):', file=self.stream)
            for hint in self.hints:
                print(f'  {hint}', file=self.stream)
        print('\nSummary: ' + ', '.join(f'{self.counts[s]} {s}' for s in
                                       ('OK', 'WARN', 'FAIL', 'SKIP')), file=self.stream)
        print('No data processing, model inference or installation was performed. '
              'Processing commands, license validity and workload memory are not tested.',
              file=self.stream)
        return 1 if self.counts['FAIL'] else 0


def pip_command(*arguments):
    return ' '.join(shlex.quote(value) for value in
                    (sys.executable, '-m', 'pip', 'install') + arguments)


def check_packages(report):
    """Import each package separately so one missing dependency does not hide others."""
    install = pip_command('-r', str(ROOT / 'requirements.txt'))
    if sys.version_info < (3, 8):
        report.add('FAIL', 'Python', 'Python 3.8 or newer is required',
                   ('From the repository root: conda env create -f environment.yml',
                    'conda activate ddsurfer'))
        return {}
    report.add('OK', 'Python', f'{sys.version.split()[0]} ({sys.executable})')
    if sys.version_info >= (3, 12):
        report.add('WARN', 'Python version', 'The pinned dependencies target Python 3.8-3.11',
                   ('From the repository root: conda env create -f environment.yml',
                    'conda activate ddsurfer'))
    from importlib import metadata

    loaded = {}
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        package, expected = line.split('==', 1)
        try:
            module = importlib.import_module(MODULES.get(package, package))
        except Exception as error:
            hints = [install]
            if package in ('torch', 'torchvision'):
                hints.append('For incompatible torch/torchvision builds, reinstall the matched pair: ' +
                             pip_command('--force-reinstall', 'torch==2.4.1', 'torchvision==0.19.1'))
            report.add('FAIL', package, f'Import failed: {type(error).__name__}: {error}', hints)
            continue
        loaded[package] = module
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            report.add('WARN', package, 'Import works, but package version metadata is missing', (install,))
            continue
        if version.split('+', 1)[0] == expected:
            report.add('OK', package, f'{version}; import successful')
        else:
            report.add('WARN', package, f'{version}; repository pins {expected}', (install,))
    return loaded


def check_device(report, torch, device, precision):
    if device == 'cpu':
        if precision == 'bf16':
            report.add('FAIL', 'Precision', 'BF16 requires a supported CUDA GPU',
                       ('Use --device cpu --precision fp32.',))
        else:
            report.add('OK', 'Inference device', 'CPU / FP32 selected')
        report.add('SKIP', 'CUDA', 'Not required for CPU; use --device cuda:0 to check a GPU')
        return
    match = re.fullmatch(r'cuda(?::(\d+))?', device)
    if not match:
        report.add('FAIL', 'Inference device', f'Unsupported device: {device}',
                   ('Use --device cpu or --device cuda:0.',))
        return
    if torch is None:
        report.add('SKIP', 'CUDA', 'Install/fix PyTorch first')
        return
    fixes = (
        'Check the NVIDIA driver and visible GPUs: nvidia-smi',
        'Check GPU visibility: echo "$CUDA_VISIBLE_DEVICES" (do not change it if set by a job scheduler).',
        'For CPU inference, use --device cpu --precision fp32.',
    )
    try:
        if not torch.cuda.is_available():
            if torch.version.cuda is None:
                fixes += (
                    'PyTorch has no CUDA build. CUDA 11.8 example, with a compatible NVIDIA driver: ' +
                    pip_command('--force-reinstall', 'torch==2.4.1', 'torchvision==0.19.1',
                                '--index-url', 'https://download.pytorch.org/whl/cu118'),
                    'Other supported CUDA builds: https://pytorch.org/get-started/previous-versions/#v241',
                )
            report.add('FAIL', 'CUDA', 'CUDA is unavailable in this Python environment', fixes)
            return
        count = torch.cuda.device_count()
        index = int(match.group(1)) if match.group(1) is not None else torch.cuda.current_device()
        if index >= count:
            report.add('FAIL', 'CUDA device', f'{device} requested, but only {count} device(s) are visible',
                       ('Select a visible device, e.g. --device cuda:0.',))
            return
        properties = torch.cuda.get_device_properties(index)
        report.add('OK', 'CUDA', f'cuda:{index}, {properties.name}, '
                   f'{properties.total_memory / 1024**3:.1f} GiB total; PyTorch CUDA {torch.version.cuda}')
        if precision == 'bf16':
            with torch.cuda.device(index):
                supported = torch.cuda.is_bf16_supported()
            report.add('OK' if supported else 'FAIL', 'BF16',
                       'Supported on the selected GPU' if supported else 'Not supported on the selected GPU',
                       () if supported else ('Use --precision fp32 or select a BF16-capable GPU.',))
        else:
            report.add('OK', 'Precision', 'FP32 selected; BF16 is not required')
    except Exception as error:
        report.add('FAIL', 'CUDA', f'{type(error).__name__}: {error}', fixes)


def check_file(report, name, path, executable=False, hints=()):
    path = Path(path)
    try:
        present = path.is_file() and path.stat().st_size > 0
    except OSError as error:
        report.add('FAIL', name, f'Cannot inspect {path}: {error}', hints)
        return
    if not present:
        report.add('FAIL', name, f'Missing or empty: {path}', hints)
    elif executable and not os.access(path, os.X_OK):
        report.add('FAIL', name, f'Not executable: {path}',
                   ('If this is a trusted installation: chmod u+x ' + shlex.quote(str(path)),))
    else:
        report.add('OK', name, str(path))


def check_slicer(report):
    # Match the preprocessing path precedence and module selection.
    selected = os.environ.get('DTI_SLICER_PATH') or os.environ.get('SLICER_PATH')
    found = shutil.which('Slicer') if not selected else None
    home = Path(selected) if selected else (Path(found).parent if found else None)
    if home is None:
        report.add('FAIL', 'Slicer', 'Set SLICER_PATH or put Slicer on PATH', SLICER_INSTALL)
        return
    check_file(report, 'Slicer', home / 'Slicer', True, SLICER_INSTALL)
    versions = sorted(path for path in (home / 'lib').glob('Slicer-*') if path.is_dir())
    if not versions:
        report.add('FAIL', 'Slicer CLI layout', 'Cannot find lib/Slicer-* under the selected installation',
                   SLICER_INSTALL)
        return
    core = versions[-1] / 'cli-modules'
    for name in ('BRAINSFit', 'ResampleScalarVectorDWIVolume'):
        check_file(report, name, core / name, True, SLICER_INSTALL)
    extensions = sorted(home.glob(f'**/SlicerDMRI/lib/{versions[-1].name}/cli-modules'))
    if not extensions:
        report.add('FAIL', 'SlicerDMRI', 'No matching SlicerDMRI CLI directory found', DMRI_INSTALL)
        return
    for name in ('DWIToDTIEstimation', 'DiffusionTensorScalarMeasurements'):
        check_file(report, name, extensions[-1] / name, True, DMRI_INSTALL)


def check_freesurfer(report, args):
    home = Path(args.freesurfer_home).expanduser().resolve() if args.freesurfer_home else None
    if args.auto_mask:
        if home is None:
            report.add('FAIL', 'SynthStrip', 'Automatic masking requires FREESURFER_HOME', FS_INSTALL)
        else:
            check_file(report, 'SynthStrip', home / 'bin/mri_synthstrip', True, FS_INSTALL)
            check_file(report, 'SynthStrip weights', home / 'models/synthstrip.1.pt', hints=FS_INSTALL)
    else:
        report.add('SKIP', 'SynthStrip', 'Optional; check with --auto-mask')
    if not args.post_process:
        report.add('SKIP', 'FreeSurfer postprocessing', 'Optional; check with --post-process')
        return
    from postprocessing.environment import check_environment, configured_home

    try:
        environment = check_environment(configured_home(args), args.postprocess_hemi,
                                        args.postprocess_atlases, backend=args.postprocess_backend)
    except (OSError, ValueError, RuntimeError) as error:
        message = str(error)
        hints = POSTPROCESS_INSTALL if args.postprocess_runtime is not None or home is None else FS_INSTALL
        if 'license' in message.lower():
            hints = FS_LICENSE
        elif 'shared librar' in message.lower():
            hints += ('On Ubuntu/Debian: sudo apt-get install libgomp1 libcrypt1 libstdc++6 zlib1g',
                      'Use a compatible Linux x86_64 host, or pass --freesurfer-home with a native installation.')
        elif home is not None and 'executable' in message.lower():
            binary = home / 'bin' / message.rsplit(': ', 1)[-1]
            if binary.is_file() and not os.access(binary, os.X_OK):
                hints = ('If this is a trusted installation: chmod u+x ' + shlex.quote(str(binary)),)
        elif 'atlas file' in message.lower():
            hints += ('Install the complete FreeSurfer distribution, including subjects/fsaverage '
                      'and the requested atlas files.',)
        report.add('FAIL', 'FreeSurfer postprocessing', message, hints)
    else:
        report.add('OK', 'FreeSurfer postprocessing',
                   f"{environment['home']}; " +
                   ('local source build verified; no full installation needed' if environment['minimal_runtime']
                    else 'required tools and atlas files found'))
        report.add('OK', 'FreeSurfer license file',
                   f"{environment['license_file']} (present and nonempty; validity not tested)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', help='cpu (default), cuda or cuda:<index>')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--skip-slicer', action='store_true',
                        help='Skip Slicer/DMRI only if not checking raw-DWI preprocessing')
    parser.add_argument('--auto-mask', action='store_true', help='Also check FreeSurfer SynthStrip')
    parser.add_argument('--post-process', action='store_true', help='Also check FreeSurfer postprocessing')
    parser.add_argument('--freesurfer-home', type=Path, default=os.environ.get('FREESURFER_HOME'))
    parser.add_argument('--postprocess-runtime', type=Path, help='Use a locally compiled surface runtime.')
    parser.add_argument('--postprocess-backend', choices=('auto', 'native', 'freesurfer'), default='auto')
    parser.add_argument('--postprocess-hemi', choices=('left', 'right', 'both'), default='both')
    parser.add_argument('--postprocess-atlases', default='aparc,aparc.a2009s')
    return parser.parse_args(argv)


def main(argv=None, stream=None):
    args = parse_args(argv)
    report = Report(stream)
    print('DDSurfer environment check\n', file=report.stream)
    loaded = check_packages(report)
    check_device(report, loaded.get('torch'), args.device, args.precision)
    if args.skip_slicer:
        report.add('SKIP', 'Slicer/DMRI', '--skip-slicer selected; raw-DWI prerequisites are not verified')
    else:
        check_slicer(report)
    check_freesurfer(report, args)
    return report.finish()


if __name__ == '__main__':
    sys.exit(main())
