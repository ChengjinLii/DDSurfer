"""Checks for shared dependencies and relocated public entry points."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from inference.predict import TANet as PredictionNetwork
from model.ddsurfer import TANet
import run_ddsurfer_pipeline as pipeline

ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_network_entry_uses_shared_model_package(self):
        self.assertIs(PredictionNetwork, TANet)
        self.assertEqual(TANet.__module__, 'model.ddsurfer')
        self.assertFalse((ROOT / 'net').exists())

    def test_processing_packages_use_matching_names(self):
        for name in ('preprocessing', 'postprocessing'):
            self.assertTrue((ROOT / name / '__init__.py').is_file())
            self.assertTrue((ROOT / name / 'run.sh').is_file())
        self.assertFalse((ROOT / 'postprocess').exists())
        args = pipeline.parse_args(['--subject', 'x', '--brain-source', './native.nii.gz'])
        command = pipeline.build_freesurfer_command(args)
        self.assertEqual(Path(command[1]), ROOT / 'postprocessing/pipeline.py')

    def test_dependencies_are_shared_at_repository_root(self):
        for name in ('environment.yml', 'requirements.txt'):
            self.assertTrue((ROOT / name).is_file())
            self.assertFalse((ROOT / 'preprocessing' / name).exists())
        requirements = {
            line.split('==')[0].lower()
            for line in (ROOT / 'requirements.txt').read_text().splitlines()
            if line.strip() and not line.startswith('#')
        }
        self.assertEqual(requirements, {'torch', 'torchvision', 'numpy', 'scipy',
                                        'nibabel', 'simpleitk', 'trimesh', 'pynrrd'})
        environment = (ROOT / 'environment.yml').read_text()
        self.assertIn('name: ddsurfer', environment)
        self.assertIn('- -r requirements.txt', environment)

    def test_public_shell_entries_work_outside_repository(self):
        env = dict(os.environ, PYTHON_BIN=sys.executable, PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory() as temp:
            for script in ('run_ddsurfer_pipeline.sh', 'preprocessing/run.sh',
                           'postprocessing/run.sh'):
                with self.subTest(script=script):
                    result = subprocess.run(['bash', str(ROOT / script), '--help'],
                                            cwd=temp, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertIn('--subject', result.stdout)

    def test_preprocessing_helpers_are_kept_with_their_stage(self):
        for name in ('mask.py', 'resample.py', 'normalize.py', 'normalize_dti.py'):
            self.assertTrue((ROOT / 'preprocessing' / name).is_file())
        for name in ('skull_stripping.py', 'nifti_resample.py', 'nifti_zscore.py',
                     'create_fs_label.py', 'io.py'):
            self.assertFalse((ROOT / 'utils' / name).exists())
        self.assertFalse((ROOT / 'inference/meshes.py').exists())
        for name in ('files.py', 'surface_io.py', 'mesh.py'):
            self.assertTrue((ROOT / 'utils' / name).is_file())

    def test_postprocessing_does_not_import_inference_or_torch(self):
        check = ('import sys; sys.path.insert(0, sys.argv[1]); '
                 'import postprocessing.pipeline; '
                 'assert not any(name == "inference" or name.startswith("inference.") '
                 'for name in sys.modules); assert "torch" not in sys.modules')
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, '-I', '-B', '-c', check, str(ROOT)],
                                    cwd=temp, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_relocated_python_helpers_work_outside_repository(self):
        scripts = ('preprocessing/inputs.py', 'preprocessing/export.py', 'preprocessing/mask.py',
                   'preprocessing/resample.py', 'preprocessing/normalize.py',
                   'preprocessing/normalize_dti.py', 'tools/obj_to_stl.py',
                   'tools/stl_to_obj.py', 'tools/translate_mesh.py')
        with tempfile.TemporaryDirectory() as temp:
            for script in scripts:
                with self.subTest(script=script):
                    result = subprocess.run([sys.executable, '-I', '-B', str(ROOT / script), '--help'],
                                            cwd=temp, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertIn('usage:', result.stdout)


if __name__ == '__main__':
    unittest.main()
