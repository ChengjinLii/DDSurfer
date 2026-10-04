"""Run diffusion preprocessing, MNI surface prediction and native-space processing."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import subprocess
import shutil
import sys
from collections import deque
from contextlib import contextmanager
from threading import current_thread, main_thread
from time import perf_counter
from pathlib import Path
from typing import Iterable, List, Sequence

from preprocessing.inputs import INPUT_NAMES, resolve_inputs
from utils.files import atomic_json, sha256_file
from utils.preflight import inspect_run, check_runtime
from utils.results import result_receipt, reusable_result
from utils.stages import subject_lock

PROJECT_ROOT = Path(__file__).resolve().parent


@contextmanager
def interruptible_run():
    def stop(signum, frame):
        raise KeyboardInterrupt('Pipeline terminated')

    previous = signal.signal(signal.SIGTERM, stop) if current_thread() is main_thread() else None
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def wait_command(command, **kwargs):
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        status = process.wait()
    except BaseException:
        # Stop the whole stage before releasing the subject lock, including shell children.
        with interruptible_cleanup():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    return status


@contextmanager
def interruptible_cleanup():
    handlers = {}
    if current_thread() is main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, signal.SIG_IGN)
    try:
        yield
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


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
        status = wait_command(command, cwd=cwd, env=env)
        if status:
            raise subprocess.CalledProcessError(status, command)
    else:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w') as stream:
            stream.write('$ ' + ' '.join(shlex.quote(str(part)) for part in command) + '\n')
            stream.flush()
            status = wait_command(command, cwd=cwd, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT)
        if status:
            with log_path.open(errors='replace') as stream:
                tail = ''.join(deque(stream, maxlen=15)).rstrip()
            logging.error('%s failed; detailed log: %s\n%s', name, log_path, tail)
            raise subprocess.CalledProcessError(status, command)
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
                            help=f'Raw {name} file; supply all four together, or omit --mask with --auto-mask.')
    parser.add_argument('--auto-mask', action='store_true',
                        help='Generate a missing mask from mean b0 using FreeSurfer SynthStrip (default: off).')
    parser.add_argument('--mask-threads', type=int, default=4,
                        help='CPU thread budget for optional mask generation (default: 4).')
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
    parser.add_argument('--precision', choices=('auto', 'bf16', 'fp32'), default='fp32',
                        help='Default: fp32 on CPU/CUDA; select bf16 on supported CUDA devices '
                             'to reduce memory use, or auto to follow the model manifest.')
    parser.add_argument('--cpu-threads', type=int, default=4,
                        help='CPU thread budget for inference (default: 4).')
    parser.add_argument('--max-gpu-gib', type=float, default=24,
                        help='GPU allocation budget for inference in GiB (default: 24).')
    parser.add_argument('--checkpoint-root', type=Path, default=PROJECT_ROOT / 'weights')
    parser.add_argument('--template-dir', type=Path, default=PROJECT_ROOT / 'template')
    parser.add_argument('--save-debug', action='store_true', help='Save crop-coordinate predictions in cache; use --keep-cache to retain.')
    parser.add_argument('--post-process', '--freesurfer', dest='freesurfer', action='store_true',
                        help='Enable FreeSurfer surface postprocessing (disabled by default).')
    parser.add_argument('--keep-cache', action='store_true',
                        help='Keep DTI, preprocessed volumes and MNI meshes; otherwise remove after success.')
    parser.add_argument('--resume', action='store_true',
                        help='Reuse completed results only when inputs, models, settings and output checksums match.')
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


def build_mask_command(args, files):
    command = [sys.executable, str(PROJECT_ROOT / 'preprocessing/brain_mask.py'),
               '--output', str(subject_directory(args) / 'dti' / f'{args.subject}-brainmask.nii.gz'),
               '--work-dir', str(cache_directory(args) / 'mask'),
               '--record', str(subject_directory(args) / 'logs/mask.json'),
               '--threads', str(args.mask_threads)]
    for name in ('dwi', 'bval', 'bvec'):
        command += [f'--{name}', str(files[name])]
    if args.freesurfer_home is not None:
        command += ['--freesurfer-home', str(args.freesurfer_home)]
    return command


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
                '--template_dir', str(args.template_dir), '--precision', args.precision,
                '--cpu-threads', str(args.cpu_threads), '--max-gpu-gib', str(args.max_gpu_gib)]
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


def finalize_outputs(args, signature=None):
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
            if sha256_file(output) != native['output_sha256']:
                raise ValueError(f'Native surface does not match its export record: {output}')
            surfaces[name] = dict(file=f'ddsurfer/{name}.obj', sha256=native['output_sha256'],
                                  checkpoint=prediction['checkpoint_file'],
                                  checkpoint_sha256=prediction['checkpoint_sha256'],
                                  precision=prediction['precision'])
    summary = dict(subject=args.subject, coordinate_space='native_scanner_RAS_mm',
                   postprocess=args.freesurfer, completed=True, surfaces=surfaces)
    if signature is not None:
        summary['result'] = dict(signature=signature, **result_receipt(args))
    atomic_json(logs / 'pipeline.json', summary)


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
    for name in ('preprocess_jobs', 'mask_threads', 'cpu_threads', 'postprocess_threads'):
        if getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    if not (0 < args.max_gpu_gib < float('inf')):
        raise ValueError('--max-gpu-gib must be finite and positive')
    if args.resume and args.skip_preprocessing:
        raise ValueError('--resume verifies final results from raw inputs; do not combine it with --skip-preprocessing')
    if subject_directory(args).is_symlink():
        raise ValueError('Refusing to write through a symlinked subject directory')
    if cache_directory(args).is_symlink():
        raise ValueError('Refusing to write through a symlinked cache directory')
    for path in (subject_directory(args) / '.pipeline.lock', subject_directory(args) / 'logs'):
        if path.is_symlink():
            raise ValueError('Refusing to use a symlinked lock or log directory')

    if args.freesurfer:
        if args.predict_mode != 'all':
            raise ValueError('FreeSurfer postprocessing requires white and pial surfaces (--predict-mode all)')
        if args.freesurfer_home is None:
            raise ValueError('Set FREESURFER_HOME or pass --freesurfer-home')

    files = None
    if not args.skip_preprocessing:
        files = resolve_inputs(args.subject, args.raw_input_root, allow_missing_mask=args.auto_mask,
                               **{name: getattr(args, name) for name in INPUT_NAMES})
        for path in files.values():
            try:
                path.relative_to(cache_directory(args))
            except ValueError:
                continue
            raise ValueError('Source inputs must not be stored inside the disposable cache')
    with subject_lock(subject_directory(args) / '.pipeline.lock'), interruptible_run():
        run_subject(args, files)


def run_subject(args, files):
    logs = subject_directory(args) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    file_log = logging.FileHandler(logs / 'pipeline.log')
    file_log.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S'))
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")
    logging.getLogger().addHandler(file_log)
    try:
        source_files = dict(files) if files is not None else None
        logging.info('Checking inputs, models and run settings.')
        inspection = inspect_run(args, source_files)
        if args.resume and reusable_result(args, inspection['signature']):
            logging.info('Reusing verified completed outputs: %s', subject_directory(args))
            return
        if args.resume:
            logging.info('No matching completed result; running the pipeline.')
        check_runtime(args, source_files, inspection)
        logging.info('Preflight passed; starting subject %s.', args.subject)
        atomic_json(logs / 'pipeline.json', dict(subject=args.subject, completed=False))
        if files is not None:
            if 'mask' not in files:
                logging.info('Generating a missing brain mask with FreeSurfer SynthStrip.')
                run_command(build_mask_command(args, files),
                            log_path=cache_directory(args) / 'logs/mask.log')
                files['mask'] = subject_directory(args) / 'dti' / f'{args.subject}-brainmask.nii.gz'
            elif args.auto_mask:
                logging.info('Using the supplied brain mask; automatic masking is not needed.')
            for name, path in files.items():
                setattr(args, name, path)
            run_command(build_preprocessing_command(args), env=dict(os.environ, PYTHON_BIN=sys.executable),
                        log_path=cache_directory(args) / 'logs/preprocessing.log')
        else:
            logging.info("Skipping preprocessing as requested.")
        run_command(build_native_dti_export_command(args), log_path=cache_directory(args) / 'logs/dti.log')
        run_command(build_prediction_command(PROJECT_ROOT / 'DDSurfer_predict.py', args, 'both'),
                    log_path=cache_directory(args) / 'logs/prediction.log')
        run_command(build_native_conversion_command(args), log_path=cache_directory(args) / 'logs/native.log')
        if args.freesurfer:
            run_command(build_freesurfer_command(args), log_path=cache_directory(args) / 'logs/postprocessing.log')
        if inspection['signature'] is not None and inspect_run(args, source_files)['signature'] != inspection['signature']:
            raise RuntimeError('Inputs, model assets or settings changed during the run; results are not reusable')
        finalize_outputs(args, inspection['signature'])
        if not args.keep_cache:
            clean_cache(args)
        logging.info('Completed native outputs: %s', subject_directory(args))
    except BaseException:
        logging.exception('Pipeline failed; intermediate cache retained for diagnosis.')
        raise
    finally:
        logging.getLogger().removeHandler(file_log)
        file_log.close()


if __name__ == "__main__":
    main()
