"""Tests for raw-DWI inputs, output layout and disposable intermediate cache."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from preprocessing.inputs import ensure_record, fingerprint, resolve_inputs, validate_inputs
from preprocessing.conversion.bval_bvec_io import encode_nrrd_gradient
import run_ddsurfer_pipeline as pipeline

ROOT = Path(__file__).resolve().parents[1]


def make_bundle(root):
    root.mkdir(parents=True, exist_ok=True)
    files = {name: root / filename for name, filename in
             (('dwi', 'dwi.nii.gz'), ('bval', 'dwi.bval'), ('bvec', 'dwi.bvec'), ('mask', 'mask.nii.gz'))}
    affine = np.diag([-1.25, 1.25, 1.25, 1.])
    affine[:3, 3] = [20., -30., -10.]
    nib.save(nib.Nifti1Image(np.ones((5, 6, 7, 7), np.float32), affine), files['dwi'])
    nib.save(nib.Nifti1Image(np.ones((5, 6, 7), np.uint8), affine), files['mask'])
    np.savetxt(files['bval'], [[0., 1000., 1000., 1000., 1000., 1000., 1000.]])
    np.savetxt(files['bvec'], np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]], float).T)
    return files


class RawPipelineTests(unittest.TestCase):
    def test_input_output_defaults_and_options(self):
        args = pipeline.parse_args(['--subject', 'x'])
        self.assertEqual(args.raw_input_root, ROOT / 'inputs')
        self.assertEqual(args.output_root, ROOT / 'outputs')
        self.assertFalse(args.freesurfer)
        self.assertFalse(args.keep_cache)
        self.assertTrue(pipeline.parse_args(['--subject', 'x', '--post-process']).freesurfer)
        self.assertTrue(pipeline.parse_args(['--subject', 'x', '--keep-cache']).keep_cache)

    def test_default_subject_input_layout_and_explicit_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = make_bundle(root / 'inputs/x')
            expected = {name: path.resolve() for name, path in files.items()}
            self.assertEqual(resolve_inputs('x', root / 'inputs'), expected)
            self.assertEqual(resolve_inputs('x', **files), expected)
            self.assertEqual(validate_inputs(files)['diffusion_volumes'], 6)
            with self.assertRaisesRegex(ValueError, 'together'):
                resolve_inputs('x', dwi=files['dwi'])

    def test_ddparcel_and_hcp_layouts_with_mixed_nifti_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = make_bundle(root / 'bundle')
            for directory, names in ((root / 'parcel', ('x.nii', 'x.bval', 'x.bvec', 'x-mask.nii.gz')),
                                     (root / 'hcp/x/T1w/Diffusion', ('data.nii', 'bvals', 'bvecs', 'nodif_brain_mask.nii.gz'))):
                directory.mkdir(parents=True)
                for name, filename in zip(('dwi', 'bval', 'bvec', 'mask'), names):
                    if name == 'dwi':
                        nib.save(nib.load(files[name]), directory / filename)
                    else:
                        (directory / filename).write_bytes(files[name].read_bytes())
            for source in (root / 'parcel', root / 'hcp'):
                self.assertEqual(validate_inputs(resolve_inputs('x', source))['b0_volumes'], 1)

    def test_gradient_count_and_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = make_bundle(Path(tmp))
            np.savetxt(files['bvec'], np.loadtxt(files['bvec']).T)
            validate_inputs(files)
            np.savetxt(files['bval'], [0, 1000])
            with self.assertRaisesRegex(ValueError, 'volume count'):
                validate_inputs(files)

    def test_dti_scalar_is_not_a_raw_dwi(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = make_bundle(Path(tmp))
            image = nib.load(files['dwi'])
            nib.save(nib.Nifti1Image(np.ones(image.shape[:3], np.float32), image.affine), files['dwi'])
            with self.assertRaisesRegex(ValueError, 'not a DTI'):
                validate_inputs(files)

    def test_mask_physical_space_is_checked_not_blindly_flipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = make_bundle(Path(tmp))
            image = nib.load(files['mask']); affine = image.affine.copy(); affine[0, 3] += 5.
            nib.save(nib.Nifti1Image(np.asarray(image.dataobj), affine), files['mask'])
            with self.assertRaisesRegex(ValueError, 'physical space'):
                validate_inputs(files)

    def test_degenerate_gradient_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = make_bundle(Path(tmp))
            np.savetxt(files['bvec'], [[0,0,0]] + [[1,0,0]] * 6)
            with self.assertRaisesRegex(ValueError, 'six tensor coefficients'):
                validate_inputs(files)

    def test_zero_bvalue_cannot_become_a_diffusion_weighted_gradient(self):
        encoded = encode_nrrd_gradient(0., [1., 0., 0.], 1000.)
        np.testing.assert_array_equal(np.fromstring(encoded, sep=' '), [0., 0., 0.])

    def test_nifti_scaling_is_applied_before_tensor_input_conversion(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = np.arange(5*6*7, dtype=np.int16).reshape(5,6,7)
            image = nib.Nifti1Image(data, np.diag([-1.25,1.25,1.25,1.]))
            image.header.set_slope_inter(2., 10.)
            source = root / 'scaled.nii.gz'; nib.save(image, source)
            output = root / 'scaled.nhdr'
            subprocess.run([sys.executable, str(ROOT / 'preprocessing/conversion/nhdr_write.py'),
                            '--nifti', str(source), '--nhdr', str(output)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            converted = sitk.GetArrayFromImage(sitk.ReadImage(str(output))).transpose(2,1,0)
            np.testing.assert_array_equal(converted, nib.load(source).get_fdata(dtype=np.float32))

    def test_raw_input_records_reject_unknown_or_mismatched_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = make_bundle(root / 'inputs')
            signature = {name: fingerprint(path) for name, path in files.items()}
            record = root / 'cache/raw_inputs.json'
            ensure_record(record, signature)
            ensure_record(record, signature)
            files['bval'].write_text('changed')
            changed = dict(signature, bval=fingerprint(files['bval']))
            with self.assertRaisesRegex(ValueError, 'different raw inputs'):
                ensure_record(record, changed)
            unknown = root / 'unknown'; unknown.mkdir(); (unknown / 'FA.nii.gz').write_text('old')
            with self.assertRaisesRegex(ValueError, 'no raw-input record'):
                ensure_record(unknown / 'raw_inputs.json', signature)

    def test_main_forwards_four_files_and_cleans_cache_after_native_conversion(self):
        for keep in (False, True):
            with self.subTest(keep=keep), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                make_bundle(root / 'inputs/x')
                argv = ['--subject', 'x', '--raw-input-root', str(root / 'inputs'), '--output-root', str(root / 'outputs')]
                if keep:
                    argv += ['--keep-cache']
                args = pipeline.parse_args(argv)
                cache = pipeline.cache_directory(args); cache.mkdir(parents=True)
                (cache / 'MNI.obj').write_text('temporary')
                native = pipeline.subject_directory(args) / 'ddsurfer'; native.mkdir()
                (native / 'lh.white.obj').write_text('final')
                with patch.object(pipeline, 'run_command') as run, patch.object(pipeline, 'finalize_outputs'):
                    pipeline.main(argv)
                commands = [call[0][0] for call in run.call_args_list]
                self.assertEqual([Path(command[1]).name for command in commands],
                                 ['run.sh', 'DDSurfer_predict.py', 'native.py'])
                for flag in ('--dwi', '--bval', '--bvec', '--mask'):
                    self.assertIn(flag, commands[0])
                self.assertEqual(cache.exists(), keep)
                self.assertTrue((native / 'lh.white.obj').is_file())

    def test_failed_pipeline_keeps_cache_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); make_bundle(root / 'inputs/x')
            argv = ['--subject', 'x', '--raw-input-root', str(root / 'inputs'), '--output-root', str(root / 'outputs')]
            args = pipeline.parse_args(argv); cache = pipeline.cache_directory(args); cache.mkdir(parents=True)
            with patch.object(pipeline, 'run_command', side_effect=RuntimeError('test failure')):
                with self.assertRaisesRegex(RuntimeError, 'test failure'):
                    pipeline.main(argv)
            self.assertTrue(cache.is_dir())

    def test_cache_cleanup_cannot_follow_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); valuable = root / 'valuable'; valuable.mkdir(); (valuable / 'input').write_text('retain')
            args = pipeline.parse_args(['--subject', 'x', '--output-root', str(root / 'out')])
            cache = pipeline.cache_directory(args); cache.parent.mkdir(parents=True)
            cache.symlink_to(valuable, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlinked cache'):
                pipeline.clean_cache(args)
            self.assertTrue((valuable / 'input').is_file())

    def test_existing_dti_does_not_bypass_default_raw_input_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); dti = root / 'dti/x'; dti.mkdir(parents=True)
            for scalar in ('FractionalAnisotropy', 'MinEigenvalue', 'MidEigenvalue', 'MaxEigenvalue', 'Trace', 'MeanDiffusivity'):
                nib.save(nib.Nifti1Image(np.ones((3,3,3), np.float32), np.eye(4)), dti / f'x-dti-{scalar}-Reg.nii.gz')
            nib.save(nib.Nifti1Image(np.ones((3,3,3), np.uint8), np.eye(4)), dti / 'x-mask-Reg.nii.gz')
            (dti / 'x-b0ToAtlasT2.tfm').write_text('fixture')
            marker = root / 'called'; helper = root / 'estimate.sh'
            helper.write_text(f'#!/bin/bash\necho called > "{marker}"\nexit 19\n')
            import sys
            env = dict(os.environ, PYTHON_BIN=sys.executable, DTI_PROCESSING_SCRIPT=str(helper), SKIP_DTI_PROCESSING='0')
            result = subprocess.run(['bash', str(ROOT / 'preprocessing/run.sh'), '--subject', 'x',
                                     '--input-root', str(root / 'dti'), '--output-root', str(root / 'volumes'),
                                     '--log-dir', str(root / 'logs')], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(marker.exists())


if __name__ == '__main__':
    unittest.main()
