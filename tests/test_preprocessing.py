"""Equivalent volume processing, minimal DTI output and selective resumption."""
import argparse
from pathlib import Path
import sys
import tempfile
from threading import Lock
import time
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from preprocessing import dti, volumes
from preprocessing.cache import related_files
from preprocessing.conversion.nhdr_write import write_nhdr
from preprocessing.mask import skull_strip
from preprocessing.normalize import normalize_volume
from preprocessing.resample import resample_image_to_target_space
from test_raw_pipeline import make_bundle


class PreprocessingTests(unittest.TestCase):
    def test_batched_volumes_equal_original_steps_and_do_not_repeat_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / 'dti'; inputs.mkdir()
            affine = np.eye(4); affine[:3, 3] = [-88.29999542236328, -129., -69.]
            shape = (12, 20, 16)
            rng = np.random.default_rng(15)
            mask = (rng.random(shape) > .2).astype(np.uint8)
            mask_path = inputs / 'x-mask-Reg.nii.gz'
            nib.save(nib.Nifti1Image(mask, affine), mask_path)
            for scalar, suffix in volumes.CHANNELS:
                nib.save(nib.Nifti1Image(rng.random(shape).astype(np.float32), affine),
                         inputs / f'x-dti-{scalar}-Reg.nii.gz')
            sitk.WriteTransform(sitk.TranslationTransform(3), str(inputs / 'x-b0ToAtlasT2.tfm'))
            geometry = dict(volumes.GEOMETRY, target_size=shape)
            with patch.dict(volumes.GEOMETRY, geometry):
                volumes.main(['--subject', 'x', '--input-dir', str(inputs), '--output-dir', str(root / 'volumes'), '--minimal'])
                output = root / 'volumes'
                original = root / 'original'; original.mkdir()
                reference_mask = original / 'mask.nii.gz'
                resample_image_to_target_space(mask_path, reference_mask, **geometry)
                reference_mask_data = nib.load(reference_mask).get_fdata().astype(bool)
                for scalar, suffix in volumes.CHANNELS:
                    masked = original / f'{suffix}.masked.nii.gz'
                    reference = original / f'{suffix}.nii.gz'
                    skull_strip(inputs / f'x-dti-{scalar}-Reg.nii.gz', mask_path, masked)
                    resample_image_to_target_space(masked, reference, **geometry)
                    normalize_volume(reference, reference, mask=reference_mask_data)
                    left, right = nib.load(reference), nib.load(output / f'x-{suffix}.nii.gz')
                    np.testing.assert_array_equal(left.get_fdata(), right.get_fdata())
                    np.testing.assert_array_equal(left.affine, right.affine)
                    self.assertEqual(left.header.binaryblock, right.header.binaryblock)
                mtimes = {p: p.stat().st_mtime_ns for p in output.rglob('*.nii.gz')}
                volumes.main(['--subject', 'x', '--input-dir', str(inputs), '--output-dir', str(output), '--minimal'])
                self.assertEqual(mtimes, {p: p.stat().st_mtime_ns for p in mtimes})
                missing = output / 'x-MD.nii.gz'; missing.unlink()
                with patch('preprocessing.volumes.normalize_volume', wraps=normalize_volume) as normalize:
                    volumes.main(['--subject', 'x', '--input-dir', str(inputs), '--output-dir', str(output), '--minimal'])
                    self.assertEqual(normalize.call_count, 1)
                np.testing.assert_array_equal(nib.load(missing).get_fdata(), nib.load(original / 'MD.nii.gz').get_fdata())

    def test_minimal_dti_keeps_commands_parallelizes_only_independent_stages_and_repairs_one_map(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = make_bundle(root / 'inputs')
            slicer = root / 'slicer'; slicer.mkdir()
            dmri, core = root / 'dmri', root / 'core'
            dmri.mkdir(); core.mkdir()
            for path in (slicer / 'Slicer', dmri / 'DWIToDTIEstimation', dmri / 'DiffusionTensorScalarMeasurements',
                         core / 'BRAINSFit', core / 'ResampleScalarVectorDWIVolume'):
                path.write_text('fixture')
            folder = root / 'dti'; folder.mkdir()
            args = argparse.Namespace(subject='x', output_dir=folder, slicer_path=slicer, dmri_cli=dmri,
                                      core_cli=core, reference_image=files['mask'], mask_flip=1, jobs=2,
                                      minimal=True, **files)
            lock = Lock(); activity = [0, 0]
            def command(argv, **kwargs):
                tool = Path(argv[2]).name if argv[0] != sys.executable else Path(argv[1]).name
                if tool == 'nhdr_write.py':
                    values = {argv[i]: argv[i+1] for i in range(2, len(argv), 2)}
                    write_nhdr(values['--nifti'], values.get('--bval'), values.get('--bvec'), values['--nhdr'])
                    return
                if tool == 'normalize_dti.py':
                    sitk.WriteImage(sitk.GetImageFromArray(np.ones((5,6,7), np.float32)),
                                    argv[argv.index('--output')+1], True)
                    return
                with lock:
                    activity[0] += 1; activity[1] = max(activity)
                time.sleep(.01)
                if tool == 'BRAINSFit':
                    self.assertEqual(argv[-2:], ['--useRigid', '--useAffine'])
                    sitk.WriteTransform(sitk.TranslationTransform(3), argv[argv.index('--linearTransform')+1])
                else:
                    if tool == 'DWIToDTIEstimation':
                        self.assertEqual(argv[4], 'WLS')
                        outputs = argv[-2:]
                    else:
                        outputs = [argv[-1]]
                    for output in outputs:
                        sitk.WriteImage(sitk.GetImageFromArray(np.ones((5,6,7), np.float32)), output, True)
                with lock:
                    activity[0] -= 1
            with patch('preprocessing.dti.subprocess.run', side_effect=command) as commands:
                dti.run(args)
                self.assertEqual(activity[1], 2)
                self.assertFalse(list(folder.glob('*Trace*')))
                self.assertFalse(list(folder.glob('*NormMasked*')))
                commands.reset_mock()
                dti.run(args)
                commands.assert_not_called()
                missing = folder / 'x-dti-MinEigenvalue-Reg.nii.gz'; missing.unlink()
                dti.run(args)
                self.assertEqual(commands.call_count, 1)
                commands.reset_mock()
                args.minimal = False
                dti.run(args)
                self.assertEqual(commands.call_count, 6)
                self.assertTrue((folder / 'x-dti-Trace-Reg.nii.gz').exists())
                self.assertEqual(len(list(folder.glob('*NormMasked*'))), 4)
                commands.reset_mock()
                related_files(folder / 'x-dti-MidEigenvalue.nhdr')[1].unlink()
                dti.run(args)
                self.assertEqual(commands.call_count, 1)


if __name__ == '__main__':
    unittest.main()
