"""Opt-in masking, native geometry and safe reuse of generated masks."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from preprocessing.brain_mask import generate_mask, mean_b0, synthstrip_environment
from preprocessing.inputs import resolve_inputs, validate_mask
import run_ddsurfer_pipeline as pipeline
from test_raw_pipeline import make_bundle


class BrainMaskTests(unittest.TestCase):
    def test_missing_mask_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = make_bundle(Path(tmp) / 'x')
            files['mask'].unlink()
            with self.assertRaises(FileNotFoundError):
                resolve_inputs('x', tmp)
            trio = {name: files[name] for name in ('dwi', 'bval', 'bvec')}
            self.assertEqual(resolve_inputs('x', tmp, allow_missing_mask=True), trio)
            self.assertEqual(resolve_inputs('x', allow_missing_mask=True, **trio), trio)
            with self.assertRaisesRegex(ValueError, 'together'):
                resolve_inputs('x', **trio)
            with self.assertRaisesRegex(ValueError, 'together'):
                resolve_inputs('x', allow_missing_mask=True, dwi=files['dwi'])
            self.assertFalse(pipeline.parse_args(['--subject', 'x']).auto_mask)

    def test_discovery_prefers_a_complete_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incomplete = make_bundle(root / 'x/T1w/Diffusion')
            incomplete['mask'].unlink()
            complete = make_bundle(root / 'x')
            self.assertEqual(resolve_inputs('x', root, allow_missing_mask=True), complete)
            complete['mask'].write_bytes(b'')
            with self.assertRaisesRegex(FileNotFoundError, 'empty'):
                resolve_inputs('x', root, allow_missing_mask=True)

    def test_mean_b0_uses_all_low_b_volumes_and_preserves_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = make_bundle(root / 'inputs')
            original = nib.load(files['dwi'])
            data = np.full(original.shape[:3] + (8,), 900., np.float32)
            data[..., 0], data[..., 1] = 2., 6.
            image = nib.Nifti1Image(data, original.affine)
            image.set_qform(original.affine, 1)
            image.set_sform(original.affine, 1)
            image.header.set_xyzt_units('mm', 'sec')
            nib.save(image, files['dwi'])
            np.savetxt(files['bval'], [0., 50.] + [1000.] * 6)
            vectors = np.loadtxt(files['bvec'])
            np.savetxt(files['bvec'], np.column_stack([vectors[:, 0], vectors]))
            before = files['dwi'].read_bytes()
            out = root / 'b0.nii.gz'
            self.assertEqual(mean_b0(files, out), 2)
            saved = nib.load(out)
            np.testing.assert_array_equal(np.asarray(saved.dataobj), 4.)
            np.testing.assert_allclose(saved.affine, image.affine)
            self.assertEqual(saved.header.get_xyzt_units(), ('mm', 'sec'))
            self.assertEqual(int(saved.header['qform_code']), 1)
            self.assertEqual(int(saved.header['sform_code']), 1)
            self.assertEqual(files['dwi'].read_bytes(), before)

    def test_generation_is_binary_native_and_reused_without_freesurfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = make_bundle(root / 'inputs')
            output, work, record = root / 'out/mask.nii.gz', root / 'cache', root / 'logs/mask.json'
            tool = root / 'mri_synthstrip'; tool.write_text('test executable')

            def synthstrip(command, **kwargs):
                self.assertNotIn('-g', command)
                image = nib.load(command[command.index('-i') + 1])
                nib.save(nib.Nifti1Image(np.ones(image.shape, np.uint8), image.affine),
                         command[command.index('-m') + 1])

            with patch('preprocessing.brain_mask.synthstrip_environment', return_value=(tool, {})), \
                    patch('preprocessing.brain_mask.subprocess.run', side_effect=synthstrip) as run:
                generate_mask(files, output, work, record)
                self.assertEqual(run.call_count, 1)
            validate_mask(files['dwi'], output)
            self.assertEqual(nib.load(output).get_data_dtype(), np.dtype('uint8'))
            with patch('preprocessing.brain_mask.synthstrip_environment') as environment:
                generate_mask(files, output, work, record)
                environment.assert_not_called()
            self.assertEqual(json.loads(record.read_text())['b0_volumes'], 1)
            old_bvals = files['bval'].read_bytes()
            np.savetxt(files['bval'], [0.] + [1200.] * 6)
            with self.assertRaisesRegex(ValueError, 'different inputs'):
                generate_mask(files, output, work, record)
            files['bval'].write_bytes(old_bvals)
            output.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'has changed'):
                generate_mask(files, output, work, record)

    def test_bad_outputs_are_not_published(self):
        for failure in ('empty', 'wrong_grid', 'nonbinary', 'nonfinite'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); files = make_bundle(root / 'inputs')
                tool = root / 'tool'; tool.write_text('test')
                output, record = root / 'mask.nii.gz', root / 'mask.json'

                def synthstrip(command, **kwargs):
                    source = nib.load(files['dwi'])
                    data = np.ones(source.shape[:3], np.float32)
                    affine = source.affine.copy()
                    if failure == 'empty': data[:] = 0
                    if failure == 'nonbinary': data[:] = .5
                    if failure == 'nonfinite': data[0, 0, 0] = np.nan
                    if failure == 'wrong_grid': affine[0, 3] += 5
                    nib.save(nib.Nifti1Image(data, affine), command[command.index('-m') + 1])

                with patch('preprocessing.brain_mask.synthstrip_environment', return_value=(tool, {})), \
                        patch('preprocessing.brain_mask.subprocess.run', side_effect=synthstrip):
                    with self.assertRaises(ValueError):
                        generate_mask(files, output, root / 'work', record)
                self.assertFalse(output.exists())
                self.assertFalse(record.exists())

    def test_unknown_existing_mask_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = make_bundle(root / 'inputs')
            before = files['mask'].read_bytes()
            output = root / 'existing.nii.gz'
            output.write_bytes(before)
            with self.assertRaisesRegex(ValueError, 'no generation record'):
                generate_mask(files, output, root / 'work', root / 'mask.json')
            self.assertEqual(before, output.read_bytes())
            with self.assertRaisesRegex(ValueError, 'overwrite a source input'):
                generate_mask(files, files['mask'], root / 'work', root / 'mask.json')
            self.assertEqual(before, files['mask'].read_bytes())

    def test_missing_executable_has_actionable_error(self):
        with patch.dict(os.environ, {}, clear=True), patch('shutil.which', return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, 'requires FreeSurfer'):
                synthstrip_environment(None, 4)

    def test_invalid_gradients_fail_before_mask_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = make_bundle(root / 'inputs')
            np.savetxt(files['bval'], [1000.] * 7)
            with patch('preprocessing.brain_mask.subprocess.run') as run:
                with self.assertRaisesRegex(ValueError, 'b0 volume'):
                    generate_mask(files, root / 'mask.nii.gz', root / 'cache', root / 'mask.json')
                run.assert_not_called()

    def test_mean_b0_cannot_overwrite_the_raw_dwi(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = make_bundle(Path(tmp))
            before = files['dwi'].read_bytes()
            with self.assertRaisesRegex(ValueError, 'overwrite a source input'):
                mean_b0(files, files['dwi'])
            self.assertEqual(before, files['dwi'].read_bytes())

    def test_pipeline_generates_mask_only_when_missing_and_retains_it(self):
        for provided in (True, False):
            with self.subTest(provided=provided), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); files = make_bundle(root / 'inputs/x')
                if not provided: files['mask'].unlink()
                argv = ['--subject', 'x', '--auto-mask', '--raw-input-root', str(root / 'inputs'),
                        '--output-root', str(root / 'outputs')]
                def run(command, **kwargs):
                    if Path(command[1]).name == 'brain_mask.py':
                        target = Path(command[command.index('--output') + 1])
                        target.parent.mkdir(parents=True)
                        source = nib.load(files['dwi'])
                        nib.save(nib.Nifti1Image(np.ones(source.shape[:3], np.uint8), source.affine), target)
                with patch.object(pipeline, 'run_command', side_effect=run) as commands, \
                        patch.object(pipeline, 'finalize_outputs'), \
                        patch.object(pipeline, 'inspect_run', return_value={'signature': None}), \
                        patch.object(pipeline, 'check_runtime'):
                    pipeline.main(argv)
                values = [call.args[0] for call in commands.call_args_list]
                names = [Path(command[1]).name for command in values]
                self.assertEqual(names.count('brain_mask.py'), 0 if provided else 1)
                self.assertNotIn('postprocessing/pipeline.py', [command[1] for command in values])
                preprocess = values[names.index('run.sh')]
                mask = Path(preprocess[preprocess.index('--mask') + 1])
                self.assertEqual(mask, files['mask'] if provided else root / 'outputs/x/dti/x-brainmask.nii.gz')
                self.assertTrue(mask.is_file())
                self.assertFalse((root / 'outputs/x/.cache').exists())


if __name__ == '__main__':
    unittest.main()
