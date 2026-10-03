"""Run diffusion preprocessing, MNI surface prediction and native-space processing."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import shutil
import sys
from collections import deque
from time import perf_counter
from pathlib import Path
from typing import Iterable, List, Sequence

from preprocessing.inputs import INPUT_NAMES, resolve_inputs
from utils.files import atomic_json

PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(command: Sequence[str], *, cwd: Path | None = None, env: dict | None = None,
                log_path: Path | None = None) -> None:
    """Keep the run log concise and write full command output to the cache."""
    script = Path(command[1] if len(command) > 1 else command[0])
    try:
        name = str(script.relative_to(PROJECT_ROOT))
    except ValueError:
        name = script.name
    logging.info('Running %s', name)
    logging.debug('Executing: %s', ' '.join(shlex.quote(str(part)) for part in command))
    started = perf_counter()
    if log_path is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    else:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w') as stream:
            stream.write('$ ' + ' '.join(shlex.quote(str(part)) for part in command) + '\n')
            stream.flush()
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            with log_path.open(errors='replace') as stream:
                tail = ''.join(deque(stream, maxlen=15)).rstrip()
            logging.error('%s failed; detailed log: %s\n%s', name, log_path, tail)
            raise subprocess.CalledProcessError(result.returncode, command)
    logging.info('Completed %s (%.1fs)', name, perf_counter() - started)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DDSurfer pipeline for a single subject.")
    parser.add_argument("--subject", required=True, help="Subject identifier, matching directory names in the input tree.")
    parser.add_argument(
        "--data-type",
        default="hcp",
        help="Dataset type (passed through to the prediction scripts).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device to use for prediction (e.g. cuda:0 or cpu).",
    )
    parser.add_argument(
        "--raw-input-root",
        type=Path,
        default=PROJECT_ROOT / "inputs",
        help="Raw DWI input directory.",
    )
    for name in INPUT_NAMES:
        parser.add_argument(f'--{name}', type=Path,
                            help=f'Raw {name} file; supply --dwi, --bval, --bvec and --mask together.')
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Final native surfaces and optional postprocess outputs under <subject>/.",
    )
    parser.add_argument(
        "--predict-mode",
        choices=("wm", "all"),
        default="all",
        help="Predict white surfaces only (wm), or white and pial surfaces (all).",
    )
    parser.add_argument('--precision', choices=('auto', 'bf16', 'fp32'), default='auto',
                        help='Inference precision: auto, bf16, or fp32.')
    parser.add_argument('--checkpoint-root', type=Path, default=PROJECT_ROOT / 'weights')
    parser.add_argument('--template-dir', type=Path, default=PROJECT_ROOT / 'template')
    parser.add_argument('--save-debug', action='store_true', help='Save crop-coordinate predictions in cache; use --keep-cache to retain.')
    parser.add_argument('--post-process', '--freesurfer', dest='freesurfer', action='store_true',
                        help='Enable FreeSurfer surface postprocessing (disabled by default).')
    parser.add_argument('--keep-cache', action='store_true',
                        help='Keep DTI, preprocessed volumes and MNI meshes; otherwise remove after success.')
    parser.add_argument('--freesurfer-home', type=Path, default=os.environ.get('FREESURFER_HOME'))
    parser.add_argument('--postprocess-hemi', choices=('left', 'right', 'both'), default='both')
    parser.add_argument('--postprocess-atlases', default='aparc,aparc.a2009s')
    parser.add_argument('--brain-source', type=Path, help='Native-space MRI; defaults to the native b0 from DTI estimation.')
    parser.add_argument('--postprocess-threads', type=int, default=4)
    parser.add_argument('--preprocess-jobs', type=int, default=1,
                        help='Concurrent independent DTI stages; registration settings stay unchanged.')
    parser.add_argument('--postprocess-serial', action='store_true', help='Process hemispheres sequentially.')
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Reuse preprocessing in this subject's retained cache (--keep-cache).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def subject_directory(args):
    return args.output_root.resolve() / args.subject


def cache_directory(args):
    return subject_directory(args) / '.cache'


def build_preprocessing_command(args: argparse.Namespace) -> List[str]:
    files = resolve_inputs(args.subject, args.raw_input_root,
                           **{name: getattr(args, name) for name in INPUT_NAMES})
    command = [
        "bash",
        str(PROJECT_ROOT / "preprocessing/run.sh"),
        "--subject",
        args.subject,
        "--raw-input-root",
        str(args.raw_input_root),
        "--input-root",
        str(cache_directory(args) / 'dti'),
        "--output-root",
        str(cache_directory(args) / 'volumes'),
        '--minimal',
        '--jobs', str(args.preprocess_jobs),
        '--log-dir', str(cache_directory(args) / 'logs/preprocessing'),
    ]
    for name in INPUT_NAMES:
        command += [f'--{name}', str(files[name])]
    return command


def build_prediction_command(
    script_path: Path,
    args: argparse.Namespace,
    hemisphere: str,
) -> List[str]:
    command = [
        sys.executable,
        str(script_path),
        "--data_type",
        args.data_type,
        "--surf_hemi",
        hemisphere,
        "--device",
        args.device,
        "--input_root",
        str(cache_directory(args) / 'volumes'),
        "--output_dir",
        str(cache_directory(args) / 'predictions'),
        "--predict_mode",
        args.predict_mode,
        "--subjects",
        args.subject,
    ]
    command += ['--checkpoint_root', str(args.checkpoint_root),
                '--template_dir', str(args.template_dir), '--precision', args.precision]
    if args.save_debug:
        command += ['--save-debug']
    return command


def build_native_conversion_command(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable, str(PROJECT_ROOT / 'inference/native.py'),
        '--subject', args.subject, '--data-root', str(cache_directory(args) / 'volumes'),
        '--pred-root', str(cache_directory(args) / 'predictions'), '--predict-mode', args.predict_mode,
        '--output-dir', str(subject_directory(args) / 'ddsurfer'),
        '--metadata-dir', str(cache_directory(args) / 'logs/native'),
    ]


def build_native_dti_export_command(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable, str(PROJECT_ROOT / 'preprocessing/export.py'),
        '--subject', args.subject,
        '--input-dir', str(cache_directory(args) / 'dti' / args.subject),
        '--output-dir', str(subject_directory(args) / 'dti'),
    ]


def native_reference(args: argparse.Namespace) -> Path:
    if args.brain_source is not None:
        return args.brain_source
    directory = cache_directory(args) / 'dti' / args.subject
    for suffix in ('.nhdr', '.nii.gz', '.nrrd', '.nii'):
        path = directory / f'{args.subject}-b0{suffix}'
        if path.is_file():
            return path
    raise FileNotFoundError('No native b0 reference found; pass --brain-source with a native-space MRI')


def build_freesurfer_command(args: argparse.Namespace) -> List[str]:
    command = [sys.executable, str(PROJECT_ROOT / 'postprocessing/pipeline.py'),
               '--subject', args.subject,
               '--brain-source', str(native_reference(args)),
               '--output-root', str(args.output_root),
               '--hemi', args.postprocess_hemi, '--atlases', args.postprocess_atlases,
               '--threads', str(args.postprocess_threads)]
    for hemi in ('lh', 'rh'):
        for kind in ('white', 'pial'):
            command += [f'--{hemi}-{kind}', str(subject_directory(args) / 'ddsurfer' / f'{hemi}.{kind}.obj')]
    if args.freesurfer_home is not None:
        command += ['--freesurfer-home', str(args.freesurfer_home)]
    if args.postprocess_serial:
        command += ['--serial']
    return command


def finalize_outputs(args):
    directory = subject_directory(args)
    logs = directory / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    record = cache_directory(args) / 'dti' / args.subject / 'raw_inputs.json'
    if record.is_file():
        shutil.copyfile(record, logs / 'raw_inputs.json')
    surfaces = {}
    kinds = ('wm', 'pial') if args.predict_mode == 'all' else ('wm',)
    for hemi, side in (('left', 'lh'), ('right', 'rh')):
        for kind in kinds:
            name = f'{side}.{"white" if kind == "wm" else "pial"}'
            output = directory / 'ddsurfer' / f'{name}.obj'
            if not output.is_file() or not output.stat().st_size:
                raise RuntimeError(f'Native output missing: {output}')
            source = cache_directory(args) / 'predictions/mni' / args.subject / f'{args.subject}_predicted_{kind}_surface_{hemi}.json'
            prediction = json.loads(source.read_text())
            native = json.loads((cache_directory(args) / 'logs/native' / f'{name}.json').read_text())
            surfaces[name] = dict(file=f'ddsurfer/{name}.obj', sha256=native['output_sha256'],
                                  checkpoint=prediction['checkpoint_file'],
                                  checkpoint_sha256=prediction['checkpoint_sha256'],
                                  precision=prediction['precision'])
    atomic_json(logs / 'pipeline.json', dict(subject=args.subject, coordinate_space='native_scanner_RAS_mm',
                                           postprocess=args.freesurfer, completed=True, surfaces=surfaces))


def clean_cache(args):
    cache = cache_directory(args)
    if cache.is_symlink():
        raise ValueError('Refusing to remove a symlinked cache directory')
    if cache.exists():
        shutil.rmtree(cache)
        logging.info('Removed intermediate cache: %s', cache)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if Path(args.subject).name != args.subject or args.subject in ('', '.', '..', 'fsaverage'):
        raise ValueError('Invalid subject identifier')
    if args.preprocess_jobs < 1:
        raise ValueError('--preprocess-jobs must be positive')
    if subject_directory(args).is_symlink():
        raise ValueError('Refusing to write through a symlinked subject directory')
    if cache_directory(args).is_symlink():
        raise ValueError('Refusing to write through a symlinked cache directory')

    if args.freesurfer:
        if args.predict_mode != 'all':
            raise ValueError('FreeSurfer postprocessing requires white and pial surfaces (--predict-mode all)')
        if args.freesurfer_home is None:
            raise ValueError('Set FREESURFER_HOME or pass --freesurfer-home')

    preprocessing = None
    if not args.skip_preprocessing:
        preprocessing = build_preprocessing_command(args)
        files = resolve_inputs(args.subject, args.raw_input_root,
                               **{name: getattr(args, name) for name in INPUT_NAMES})
        for path in files.values():
            try:
                path.relative_to(cache_directory(args))
            except ValueError:
                continue
            raise ValueError('Source inputs must not be stored inside the disposable cache')
    logs = subject_directory(args) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    file_log = logging.FileHandler(logs / 'pipeline.log')
    file_log.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S'))
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")
    logging.getLogger().addHandler(file_log)
    try:
        if preprocessing is not None:
            run_command(preprocessing, env=dict(os.environ, PYTHON_BIN=sys.executable),
                        log_path=cache_directory(args) / 'logs/preprocessing.log')
        else:
            logging.info("Skipping preprocessing as requested.")
        run_command(build_native_dti_export_command(args), log_path=cache_directory(args) / 'logs/dti.log')
        run_command(build_prediction_command(PROJECT_ROOT / 'DDSurfer_predict.py', args, 'both'),
                    log_path=cache_directory(args) / 'logs/prediction.log')
        run_command(build_native_conversion_command(args), log_path=cache_directory(args) / 'logs/native.log')
        if args.freesurfer:
            run_command(build_freesurfer_command(args), log_path=cache_directory(args) / 'logs/postprocessing.log')
        finalize_outputs(args)
        if not args.keep_cache:
            clean_cache(args)
        logging.info('Completed native outputs: %s', subject_directory(args))
    except Exception:
        logging.exception('Pipeline failed; intermediate cache retained for diagnosis.')
        raise
    finally:
        logging.getLogger().removeHandler(file_log)
        file_log.close()


if __name__ == "__main__":
    main()
