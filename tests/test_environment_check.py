"""Environment diagnostics should report actionable failures without workloads."""

from contextlib import nullcontext
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import check_environment as diagnostic


def make_slicer(root):
    home = root / 'slicer'
    core = home / 'lib/Slicer-5.10/cli-modules'
    dmri = home / 'extensions/SlicerDMRI/lib/Slicer-5.10/cli-modules'
    for path in ([home / 'Slicer'] + [core / name for name in ('BRAINSFit', 'ResampleScalarVectorDWIVolume')]
                 + [dmri / name for name in ('DWIToDTIEstimation', 'DiffusionTensorScalarMeasurements')]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#!/bin/sh\nexit 0\n')
        path.chmod(0o755)
    return home


def make_freesurfer(root):
    home = root / 'freesurfer'
    names = ('mris_curvature', 'mris_inflate', 'mris_sphere', 'mris_register', 'mris_place_surface',
             'mri_label2label', 'mri_surf2surf', 'mri_synthstrip')
    for name in names:
        path = home / 'bin' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#!/bin/sh\nexit 0\n')
        path.chmod(0o755)
    files = [home / 'license.txt', home / 'models/synthstrip.1.pt']
    for hemi in ('lh', 'rh'):
        files += [home / 'subjects/fsaverage/label' / f'{hemi}.cortex.label',
                  home / 'subjects/fsaverage/surf' / f'{hemi}.sphere.reg',
                  home / 'average' / f'{hemi}.average.curvature.filled.buckner40.tif']
        files += [home / 'subjects/fsaverage/label' / f'{hemi}.{atlas}.annot'
                  for atlas in ('aparc', 'aparc.a2009s')]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture\n')
    return home


class EnvironmentCheckTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.report = diagnostic.Report(self.output)

    def test_help_works_without_site_packages_outside_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-I', '-S', '-B',
                                     str(diagnostic.ROOT / 'tests/check_environment.py'), '--help'],
                                    cwd=tmp, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--post-process', result.stdout)

    def test_missing_packages_are_reported_without_tracebacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-I', '-S', '-B',
                                     str(diagnostic.ROOT / 'tests/check_environment.py'), '--skip-slicer'],
                                    cwd=tmp, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        for name in ('torch', 'torchvision', 'numpy', 'scipy', 'nibabel', 'SimpleITK', 'trimesh', 'pynrrd'):
            self.assertIn(f'[FAIL] {name}:', result.stdout)
        self.assertIn('-m pip install -r', result.stdout)
        self.assertNotIn('Traceback', result.stderr)

    def test_package_versions_allow_cuda_suffix_and_map_pynrrd(self):
        from importlib import metadata
        pins = dict(line.split('==') for line in (diagnostic.ROOT / 'requirements.txt').read_text().splitlines())
        pins['torch'] += '+cu118'
        with patch.object(diagnostic.importlib, 'import_module', return_value=Mock()) as load, \
                patch.object(metadata, 'version', side_effect=pins.__getitem__):
            loaded = diagnostic.check_packages(self.report)
        self.assertIn('pynrrd', loaded)
        load.assert_any_call('nrrd')
        self.assertEqual(self.report.finish(), 0)
        self.assertEqual(self.report.counts['WARN'], 0)

    def test_broken_package_imports_do_not_stop_remaining_checks(self):
        with patch.object(diagnostic.importlib, 'import_module', side_effect=RuntimeError('ABI mismatch')):
            self.assertEqual(diagnostic.check_packages(self.report), {})
        self.assertEqual(self.report.counts['FAIL'], 8)
        self.assertIn('ABI mismatch', self.output.getvalue())
        self.report.finish()
        self.assertIn('--force-reinstall torch==2.4.1 torchvision==0.19.1', self.output.getvalue())

    def test_nonmatching_package_versions_are_warnings(self):
        from importlib import metadata
        with patch.object(diagnostic.importlib, 'import_module', return_value=Mock()), \
                patch.object(metadata, 'version', return_value='0.0'):
            self.assertEqual(len(diagnostic.check_packages(self.report)), 8)
        self.assertEqual(self.report.counts['WARN'], 8)
        self.assertEqual(self.report.finish(), 0)
        self.assertIn('-m pip install -r', self.output.getvalue())

    def test_missing_version_metadata_does_not_hide_import_success(self):
        from importlib import metadata
        with patch.object(diagnostic.importlib, 'import_module', return_value=Mock()), \
                patch.object(metadata, 'version', side_effect=metadata.PackageNotFoundError):
            self.assertEqual(len(diagnostic.check_packages(self.report)), 8)
        self.assertEqual(self.report.counts['WARN'], 8)
        self.assertEqual(self.report.finish(), 0)

    def test_cpu_does_not_inspect_cuda_and_rejects_bf16(self):
        torch = Mock()
        diagnostic.check_device(self.report, torch, 'cpu', 'fp32')
        torch.cuda.is_available.assert_not_called()
        self.assertEqual(self.report.finish(), 0)
        diagnostic.check_device(self.report, torch, 'cpu', 'bf16')
        self.assertEqual(self.report.finish(), 1)

    def test_unavailable_cuda_build_has_install_hint(self):
        torch = SimpleNamespace(cuda=Mock(), version=SimpleNamespace(cuda=None))
        torch.cuda.is_available.return_value = False
        diagnostic.check_device(self.report, torch, 'cuda:0', 'fp32')
        self.assertEqual(self.report.finish(), 1)
        self.assertIn('https://download.pytorch.org/whl/cu118', self.output.getvalue())
        self.assertIn('nvidia-smi', self.output.getvalue())

    def test_invalid_device_does_not_allocate_tensors(self):
        torch = Mock()
        for device in ('mps', 'cpu:1', 'cuda:-1'):
            diagnostic.check_device(self.report, torch, device, 'fp32')
        self.assertEqual(self.report.counts['FAIL'], 3)
        torch.cuda.is_available.assert_not_called()
        torch.empty.assert_not_called()

    def test_selected_cuda_index_and_bf16_capability(self):
        torch = SimpleNamespace(cuda=Mock(), version=SimpleNamespace(cuda='11.8'))
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 2
        torch.cuda.get_device_properties.return_value = SimpleNamespace(name='fixture GPU', total_memory=24 * 1024**3)
        torch.cuda.device.side_effect = lambda index: nullcontext()
        torch.cuda.is_bf16_supported.return_value = False
        diagnostic.check_device(self.report, torch, 'cuda:2', 'fp32')
        diagnostic.check_device(self.report, torch, 'cuda:1', 'bf16')
        self.assertEqual(self.report.counts['FAIL'], 2)
        torch.cuda.device.assert_called_once_with(1)
        torch.cuda.is_bf16_supported.return_value = True
        report = diagnostic.Report(self.output)
        diagnostic.check_device(report, torch, 'cuda:1', 'bf16')
        self.assertEqual(report.finish(), 0)

    def test_optional_tools_are_skipped_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            diagnostic.check_freesurfer(self.report, diagnostic.parse_args([]))
        self.assertEqual(self.report.counts['SKIP'], 2)
        self.assertEqual(self.report.finish(), 0)

    def test_missing_requested_freesurfer_has_setup_commands(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch('postprocessing.environment.check_environment', side_effect=FileNotFoundError('Set FS_LICENSE')):
            diagnostic.check_freesurfer(self.report, diagnostic.parse_args(['--post-process', '--auto-mask']))
        self.assertEqual(self.report.counts['FAIL'], 2)
        self.assertEqual(self.report.finish(), 1)
        self.assertIn('export FREESURFER_HOME=', self.output.getvalue())
        self.assertIn('SetUpFreeSurfer.sh', self.output.getvalue())

    def test_local_build_needs_no_configured_full_installation(self):
        environment = dict(home=Path('cache/runtime'), license_file=Path('private/license.txt'), minimal_runtime=True)
        with patch.dict(os.environ, {}, clear=True), \
                patch('postprocessing.environment.check_environment', return_value=environment) as check:
            diagnostic.check_freesurfer(self.report, diagnostic.parse_args(['--post-process']))
        check.assert_called_once_with(None, 'both', 'aparc,aparc.a2009s', backend='auto')
        self.assertEqual(self.report.finish(), 0)
        self.assertIn('no full installation needed', self.output.getvalue())

    def test_explicit_postprocessing_runtime_does_not_replace_synthstrip_home(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            home = make_freesurfer(Path(tmp))
            runtime = Path(tmp) / 'runtime'
            environment = dict(home=runtime, license_file=home / 'license.txt', minimal_runtime=True)
            args = diagnostic.parse_args(['--post-process', '--auto-mask', '--freesurfer-home', str(home),
                                          '--postprocess-runtime', str(runtime)])
            with patch('postprocessing.environment.check_environment', return_value=environment) as check:
                diagnostic.check_freesurfer(self.report, args)
            check.assert_called_once_with(runtime, 'both', 'aparc,aparc.a2009s', backend='auto')
            self.assertEqual(self.report.finish(), 0)

    def test_missing_slicer_has_configuration_hints(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(diagnostic.shutil, 'which', return_value=None):
            diagnostic.check_slicer(self.report)
        self.assertEqual(self.report.finish(), 1)
        self.assertIn('export SLICER_PATH=', self.output.getvalue())

    def test_slicer_matches_preflight_and_dti_path_precedence(self):
        from utils.preflight import slicer_files
        with tempfile.TemporaryDirectory() as tmp:
            home = make_slicer(Path(tmp))
            with patch.dict(os.environ, {'SLICER_PATH': '/missing', 'DTI_SLICER_PATH': str(home)}, clear=True):
                diagnostic.check_slicer(self.report)
                for path in slicer_files():
                    self.assertIn(str(path), self.output.getvalue())
        self.assertEqual(self.report.finish(), 0)

    def test_missing_dmri_and_nonexecutable_modules_have_distinct_fixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = make_slicer(Path(tmp))
            (home / 'extensions/SlicerDMRI/lib/Slicer-5.10/cli-modules/DWIToDTIEstimation').unlink()
            (home / 'lib/Slicer-5.10/cli-modules/BRAINSFit').chmod(0o644)
            with patch.dict(os.environ, {'SLICER_PATH': str(home)}, clear=True):
                diagnostic.check_slicer(self.report)
        self.assertEqual(self.report.counts['FAIL'], 2)
        self.report.finish()
        self.assertIn('Extensions Manager', self.output.getvalue())
        self.assertIn('chmod u+x', self.output.getvalue())

    def test_unreadable_file_is_a_failure_not_a_traceback(self):
        with patch.object(Path, 'is_file', side_effect=PermissionError('permission denied')):
            diagnostic.check_file(self.report, 'fixture', Path('inaccessible'))
        self.assertEqual(self.report.finish(), 1)
        self.assertIn('permission denied', self.output.getvalue())

    def test_freesurfer_license_atlas_and_mask_checks(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            home = make_freesurfer(Path(tmp))
            args = diagnostic.parse_args(['--freesurfer-home', str(home), '--post-process', '--auto-mask'])
            diagnostic.check_freesurfer(self.report, args)
            self.assertEqual(self.report.finish(), 0)
            (home / 'license.txt').unlink()
            report = diagnostic.Report(self.output)
            diagnostic.check_freesurfer(report, args)
            self.assertEqual(report.finish(), 1)
            self.assertIn('export FS_LICENSE=', self.output.getvalue())
            (home / 'license.txt').write_text('fixture\n')
            (home / 'subjects/fsaverage/label/rh.aparc.annot').unlink()
            report = diagnostic.Report(self.output)
            diagnostic.check_freesurfer(report, args)
            self.assertEqual(report.finish(), 1)
            self.assertIn('including subjects/fsaverage', self.output.getvalue())

    def test_mask_only_does_not_require_license_or_atlases(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            home = make_freesurfer(Path(tmp))
            (home / 'license.txt').unlink()
            args = diagnostic.parse_args(['--freesurfer-home', str(home), '--auto-mask'])
            diagnostic.check_freesurfer(self.report, args)
            self.assertEqual(self.report.finish(), 0)
            (home / 'models/synthstrip.1.pt').unlink()
            diagnostic.check_freesurfer(self.report, args)
            self.assertEqual(self.report.finish(), 1)

    def test_main_skip_option_and_exit_status(self):
        with patch.object(diagnostic, 'check_packages', return_value={}), \
                patch.object(diagnostic, 'check_slicer') as slicer, patch.dict(os.environ, {}, clear=True):
            status = diagnostic.main(['--skip-slicer'], stream=self.output)
        self.assertEqual(status, 0)
        slicer.assert_not_called()
        self.assertNotIn('Suggested fixes', self.output.getvalue())
        self.assertIn('No data processing', self.output.getvalue())


if __name__ == '__main__':
    unittest.main()
