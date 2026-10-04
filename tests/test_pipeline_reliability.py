"""Early validation, subject isolation and reuse after cache cleanup."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

import run_ddsurfer_pipeline as pipeline
from test_raw_pipeline import make_bundle
from utils.files import atomic_json, sha256_file
from utils.preflight import check_device, check_runtime, inspect_run, slicer_files
from utils.results import result_receipt, reusable_result
from utils.stages import subject_lock

ROOT = Path(__file__).resolve().parents[1]


def make_assets(root):
    folder = root / 'weights'
    folder.mkdir()
    config = json.loads((ROOT / 'weights/manifest.json').read_text())
    entries = []
    for hemi in ('left', 'right'):
        for kind, name in config['checkpoint_files'][hemi].items():
            path = folder / name
            path.write_bytes(f'{hemi}-{kind}-fixture'.encode())
            entries.append(f'{sha256_file(path)}  {name}\n')
    atomic_json(folder / 'manifest.json', config)
    (folder / 'SHA256SUMS').write_text(''.join(entries))
    return folder


def make_slicer(root):
    home = root / 'slicer'
    core = home / 'lib/Slicer-5.2/cli-modules'
    dmri = home / 'extensions/SlicerDMRI/lib/Slicer-5.2/cli-modules'
    core.mkdir(parents=True); dmri.mkdir(parents=True)
    for path in ([home / 'Slicer'] + [core / n for n in ('BRAINSFit', 'ResampleScalarVectorDWIVolume')]
                 + [dmri / n for n in ('DWIToDTIEstimation', 'DiffusionTensorScalarMeasurements')]):
        path.write_text('#!/bin/sh\nexit 0\n'); path.chmod(0o755)
    return home


def make_results(args):
    root = pipeline.subject_directory(args)
    for folder in ('dti', 'ddsurfer', 'logs'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for name in ('FA', 'MD', 'MinEigenvalue', 'MidEigenvalue', 'MaxEigenvalue', 'Trace'):
        (root / 'dti' / f'x-{name}.nii.gz').write_text(name)
    for side in ('lh', 'rh'):
        for kind in ('white', 'pial'):
            (root / 'ddsurfer' / f'{side}.{kind}.obj').write_text(f'{side}.{kind}')
    for name in ('raw_inputs.json', 'dti.json', 'b0ToAtlasT2.tfm'):
        (root / 'logs' / name).write_text('{}')


def write_receipt(args, signature):
    atomic_json(pipeline.subject_directory(args) / 'logs/pipeline.json',
                dict(subject=args.subject, completed=True,
                     result=dict(signature=signature, **result_receipt(args))))


class PreflightTests(unittest.TestCase):
    def test_resource_limits_are_forwarded(self):
        args = pipeline.parse_args(['--subject', 'x', '--cpu-threads', '7', '--max-gpu-gib', '18'])
        command = pipeline.build_prediction_command(ROOT / 'DDSurfer_predict.py', args, 'both')
        self.assertEqual(command[command.index('--cpu-threads') + 1], '7')
        self.assertEqual(command[command.index('--max-gpu-gib') + 1], '18.0')
        self.assertEqual(pipeline.parse_args(['--subject', 'x']).precision, 'fp32')

    def test_invalid_limits_fail_before_any_stage(self):
        for flag, value in (('--cpu-threads', '0'), ('--max-gpu-gib', 'nan'),
                            ('--max-gpu-gib', '-1'), ('--postprocess-threads', '0')):
            with self.subTest(flag=flag, value=value), patch.object(pipeline, 'run_command') as run:
                with self.assertRaises(ValueError):
                    pipeline.main(['--subject', 'x', flag, value])
                run.assert_not_called()

    def test_cpu_precision_and_invalid_devices(self):
        config = {'precision': 'fp32'}
        args = pipeline.parse_args(['--subject', 'x', '--device', 'cpu'])
        self.assertEqual(check_device(args, config), 'fp32')
        args.precision = 'bf16'
        with self.assertRaisesRegex(ValueError, 'BF16'):
            check_device(args, config)
        for name in ('mps', 'cpu:1'):
            args.device = name
            with self.assertRaisesRegex(ValueError, 'device cpu'):
                check_device(args, config)

    def test_unavailable_and_invalid_cuda(self):
        args = pipeline.parse_args(['--subject', 'x'])
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                check_device(args, {'precision': 'fp32'})
        args.device = 'cuda:3'
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2):
            with self.assertRaisesRegex(ValueError, 'index out of range'):
                check_device(args, {'precision': 'fp32'})

    def test_cuda_precision_and_memory_checks_use_selected_device(self):
        args = pipeline.parse_args(['--subject', 'x', '--device', 'cuda:1', '--precision', 'bf16'])
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2), \
                patch('torch.cuda.device', return_value=nullcontext()) as device, \
                patch('torch.cuda.is_bf16_supported', return_value=False):
            with self.assertRaisesRegex(ValueError, 'BF16'):
                check_device(args, {'precision': 'fp32'})
            device.assert_called_once_with(1)
        args.precision = 'fp32'
        for free, cap in ((11, 24), (24, 7)):
            args.max_gpu_gib = cap
            with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2), \
                    patch('torch.cuda.device', return_value=nullcontext()), \
                    patch('torch.cuda.mem_get_info', return_value=(free * 1024**3, 80 * 1024**3)):
                with self.assertRaisesRegex(RuntimeError, '8-GiB'):
                    check_device(args, {'precision': 'fp32'})

    def test_slicer_layout_and_missing_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = make_slicer(Path(tmp))
            with patch.dict(os.environ, {'DTI_SLICER_PATH': str(home)}):
                files = slicer_files()
                self.assertEqual(len(files), 5)
                next(p for p in files if p.name == 'BRAINSFit').chmod(0o644)
                with self.assertRaisesRegex(FileNotFoundError, 'BRAINSFit'):
                    slicer_files()
        with patch.dict(os.environ, {}, clear=True), patch('shutil.which', return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, 'SLICER_PATH'):
                slicer_files()

    def test_signature_checks_inputs_assets_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = make_bundle(root / 'inputs/x')
            weights = make_assets(root)
            args = pipeline.parse_args(['--subject', 'x', '--device', 'cpu', '--checkpoint-root', str(weights)])
            with patch('utils.preflight.slicer_files', return_value=[]):
                clean = inspect_run(args, files)['signature']
                args.resume = True; args.keep_cache = True; args.log_level = 'DEBUG'
                self.assertEqual(inspect_run(args, files)['signature'], clean)
                args.cpu_threads = 3
                self.assertNotEqual(inspect_run(args, files)['signature'], clean)
                args.cpu_threads = 4
                files['bval'].write_text('0 1100 1000 1000 1000 1000 1000\n')
                self.assertNotEqual(inspect_run(args, files)['signature'], clean)
                (weights / 'ddsurfer_lh_wm.pt').write_text('corrupt')
                with self.assertRaisesRegex(ValueError, 'Checkpoint checksum'):
                    inspect_run(args, files)

    def test_template_checksum_fails_before_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = make_bundle(root / 'inputs/x'); weights = make_assets(root)
            templates = root / 'template'; templates.mkdir()
            (templates / 'ddsurfer_lh_template.obj').write_text('corrupt')
            args = pipeline.parse_args(['--subject', 'x', '--checkpoint-root', str(weights), '--template-dir', str(templates)])
            with self.assertRaisesRegex(ValueError, 'Template checksum'):
                inspect_run(args, files)

    def test_incompatible_state_dict_is_rejected_before_slicer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.pt'; torch.save({'wrong_key': torch.ones(1)}, path)
            args = pipeline.parse_args(['--subject', 'x', '--device', 'cpu'])
            config = json.loads((ROOT / 'weights/manifest.json').read_text())
            with patch('utils.preflight.slicer_files') as slicer:
                with self.assertRaisesRegex(RuntimeError, 'state_dict'):
                    check_runtime(args, {}, dict(config=config, models=[(path, sha256_file(path))]))
                slicer.assert_not_called()

    def test_freesurfer_and_synthstrip_are_only_required_when_requested(self):
        args = pipeline.parse_args(['--subject', 'x', '--device', 'cpu'])
        inspection = dict(config={'precision': 'fp32'}, models=[])
        with patch('utils.preflight.slicer_files'), patch('postprocessing.environment.check_environment') as fs, \
                patch('preprocessing.brain_mask.synthstrip_environment') as mask:
            check_runtime(args, {'mask': Path('mask')}, inspection)
            fs.assert_not_called(); mask.assert_not_called()
        args.freesurfer = True
        with self.assertRaisesRegex(ValueError, 'FREESURFER_HOME'):
            check_runtime(args, None, inspection)
        args.freesurfer = False
        with patch('utils.preflight.slicer_files'), patch('preprocessing.brain_mask.synthstrip_environment',
                                                          side_effect=FileNotFoundError('missing synthstrip')):
            with self.assertRaisesRegex(FileNotFoundError, 'synthstrip'):
                check_runtime(args, {}, inspection)

    def test_preflight_failure_does_not_start_processing_or_invalidate_old_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); make_bundle(root / 'inputs/x')
            argv = ['--subject', 'x', '--raw-input-root', str(root / 'inputs'), '--output-root', str(root / 'outputs')]
            args = pipeline.parse_args(argv); make_results(args); write_receipt(args, 'old')
            receipt = (pipeline.subject_directory(args) / 'logs/pipeline.json').read_bytes()
            with patch.object(pipeline, 'inspect_run', return_value={'signature': 'new'}), \
                    patch.object(pipeline, 'check_runtime', side_effect=RuntimeError('GPU preflight')), \
                    patch.object(pipeline, 'run_command') as run:
                with self.assertRaisesRegex(RuntimeError, 'GPU preflight'):
                    pipeline.main(argv)
                run.assert_not_called()
            self.assertEqual((pipeline.subject_directory(args) / 'logs/pipeline.json').read_bytes(), receipt)


class ResultReuseTests(unittest.TestCase):
    def test_resume_reuses_after_cache_cleanup_without_running_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); make_bundle(root / 'inputs/x')
            argv = ['--subject', 'x', '--resume', '--raw-input-root', str(root / 'inputs'), '--output-root', str(root / 'outputs')]
            args = pipeline.parse_args(argv); make_results(args); write_receipt(args, 'match')
            summary = (pipeline.subject_directory(args) / 'logs/pipeline.json').read_bytes()
            with patch.object(pipeline, 'inspect_run', return_value={'signature': 'match'}), \
                    patch.object(pipeline, 'check_runtime') as runtime, patch.object(pipeline, 'run_command') as run:
                pipeline.main(argv)
                runtime.assert_not_called(); run.assert_not_called()
            self.assertEqual((pipeline.subject_directory(args) / 'logs/pipeline.json').read_bytes(), summary)
            self.assertFalse(pipeline.cache_directory(args).exists())

    def test_changed_incomplete_and_old_receipts_are_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(['--subject', 'x', '--output-root', tmp]); make_results(args)
            write_receipt(args, 'match')
            self.assertFalse(reusable_result(args, 'changed-input-or-setting'))
            path = pipeline.subject_directory(args) / 'logs/pipeline.json'
            for record in ({'subject': 'x', 'completed': True}, {'subject': 'x', 'completed': False},
                           {'subject': 'other', 'completed': True}, []):
                atomic_json(path, record)
                self.assertFalse(reusable_result(args, 'match'))

    def test_output_edits_removal_empty_files_and_symlinks_invalidate_reuse(self):
        for action in ('edit', 'remove', 'empty', 'symlink', 'extra'):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmp:
                args = pipeline.parse_args(['--subject', 'x', '--output-root', tmp]); make_results(args); write_receipt(args, 'match')
                path = pipeline.subject_directory(args) / 'dti/x-FA.nii.gz'
                if action == 'edit': path.write_text('edited')
                elif action == 'remove': path.unlink()
                elif action == 'empty': path.write_text('')
                elif action == 'symlink':
                    target = Path(tmp) / 'target'; target.write_bytes(path.read_bytes()); path.unlink(); path.symlink_to(target)
                else: (path.parent / 'unexpected.txt').write_text('extra')
                self.assertFalse(reusable_result(args, 'match'))

    def test_postprocess_results_are_included_in_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(['--subject', 'x', '--output-root', tmp, '--post-process',
                                        '--postprocess-hemi', 'left', '--postprocess-atlases', 'aparc'])
            make_results(args); root = pipeline.subject_directory(args)
            paths = ['mri/brain.mgz', 'mri/orig.mgz', 'label/lh.cortex.label', 'label/lh.aparc.annot', 'stats/lh.aparc.stats']
            paths += ['surf/lh.' + n for n in ('white', 'pial', 'curv', 'sulc', 'thickness', 'area', 'inflated', 'sphere', 'sphere.reg')]
            paths += ['fsaverage/lh.' + n + '.mgh' for n in ('thickness', 'curv', 'sulc', 'area.pial')]
            paths += ['label/lh.aparc/lh.region.label']
            for name in paths:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture')
            write_receipt(args, 'match')
            self.assertTrue(reusable_result(args, 'match'))
            (root / 'label/lh.aparc/lh.region.label').unlink()
            self.assertFalse(reusable_result(args, 'match'))

    def test_failed_rerun_invalidates_previous_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); make_bundle(root / 'inputs/x')
            argv = ['--subject', 'x', '--resume', '--raw-input-root', str(root / 'inputs'), '--output-root', str(root / 'outputs')]
            args = pipeline.parse_args(argv); make_results(args); write_receipt(args, 'old')
            with patch.object(pipeline, 'inspect_run', return_value={'signature': 'new'}), \
                    patch.object(pipeline, 'check_runtime'), patch.object(pipeline, 'run_command', side_effect=RuntimeError('stage failure')):
                with self.assertRaisesRegex(RuntimeError, 'stage failure'):
                    pipeline.main(argv)
            self.assertFalse(json.loads((pipeline.subject_directory(args) / 'logs/pipeline.json').read_text())['completed'])
            self.assertFalse(reusable_result(args, 'old'))


class SubjectLockTests(unittest.TestCase):
    def test_duplicate_main_is_rejected_before_logs_or_cache_are_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(['--subject', 'x', '--skip-preprocessing', '--output-root', tmp])
            with subject_lock(pipeline.subject_directory(args) / '.pipeline.lock'), patch.object(pipeline, 'inspect_run') as inspect:
                with self.assertRaisesRegex(RuntimeError, 'Another process'):
                    pipeline.main(['--subject', 'x', '--skip-preprocessing', '--output-root', tmp])
                inspect.assert_not_called()
            self.assertFalse((pipeline.subject_directory(args) / 'logs').exists())

    def test_lock_survives_cleanup_and_releases_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(['--subject', 'x', '--output-root', tmp])
            path = pipeline.subject_directory(args) / '.pipeline.lock'
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with subject_lock(path):
                    inode = path.stat().st_ino
                    pipeline.cache_directory(args).mkdir()
                    pipeline.clean_cache(args)
                    raise RuntimeError('fixture')
            with subject_lock(path):
                self.assertEqual(path.stat().st_ino, inode)

    def test_lock_file_cannot_follow_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '.lock'; target = Path(tmp) / 'target'; target.write_text('untouched')
            path.symlink_to(target)
            with self.assertRaises(OSError):
                with subject_lock(path): pass
            self.assertEqual(target.read_text(), 'untouched')

    def test_termination_stops_stage_children_before_unlocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ready = root / 'stage.pid'
            stage = ('import os,time; from pathlib import Path; '
                     f'Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(120)')
            driver = ('from unittest.mock import patch; import run_ddsurfer_pipeline as p; import sys; '
                      'original=p.run_command; '
                      f'argv=["--subject","x","--skip-preprocessing","--output-root",{tmp!r}]; '
                      'inspection=patch.object(p,"inspect_run",return_value={"signature":None}); '
                      'runtime=patch.object(p,"check_runtime"); '
                      f'run=patch.object(p,"run_command",side_effect=lambda *a,**kw: original([sys.executable,"-c",{stage!r}],**kw)); '
                      'inspection.start(); runtime.start(); run.start(); p.main(argv)')
            process = subprocess.Popen([sys.executable, '-c', driver], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 15
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue(ready.exists())
                child = int(ready.read_text())
                process.send_signal(signal.SIGTERM); process.wait(timeout=15)
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse(Path(f'/proc/{child}').exists())
                with subject_lock(root / 'x/.pipeline.lock'): pass
                self.assertFalse(json.loads((root / 'x/logs/pipeline.json').read_text())['completed'])
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()


if __name__ == '__main__':
    unittest.main()
