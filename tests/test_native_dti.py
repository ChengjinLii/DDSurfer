"""Native scalar export preserves values, image geometry and independent outputs."""
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from preprocessing.export import SCALARS, export_native


def make_native_cache(root, trace=True):
    root.mkdir(parents=True)
    shape = (5, 6, 7)
    angle = .23
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([-1.25, 1.8, 2.])
    affine[:3, 3] = [44., -28., -19.]
    geometry = dict(shape=list(shape) + [65], affine=affine.tolist())
    record = dict(geometry=geometry, inputs=dict(dwi=dict(sha256='fixture')))
    (root / 'raw_inputs.json').write_text(json.dumps(record))
    lps = np.diag([-1., -1., 1., 1.]) @ affine
    spacing = np.linalg.norm(lps[:3, :3], axis=0)
    values = {}
    for index, (scalar, name) in enumerate(SCALARS):
        if name == 'Trace' and not trace:
            continue
        array = np.linspace(-.001, .002, np.prod(shape), dtype=np.float32).reshape(shape) * (index + 1)
        image = sitk.GetImageFromArray(array.transpose(2, 1, 0).copy())
        image.SetOrigin(tuple(lps[:3, 3]))
        image.SetSpacing(tuple(spacing))
        image.SetDirection(tuple((lps[:3, :3] / spacing).ravel()))
        sitk.WriteImage(image, str(root / f'x-dti-{scalar}.nhdr'), True)
        values[name] = array
    return affine, values


class NativeDtiTests(unittest.TestCase):
    def test_unnormalized_values_and_oblique_native_physical_space_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            affine, values = make_native_cache(root / 'cache')
            result = export_native('x', root / 'cache', root / 'outputs/x/dti')
            self.assertEqual(set(result['scalar_maps']), set(values))
            self.assertFalse(result['normalized'])
            self.assertEqual(result['coordinate_space'], 'native_scanner_RAS_mm')
            for name, array in values.items():
                file = root / 'outputs/x' / result['scalar_maps'][name]['file']
                image = nib.load(file)
                np.testing.assert_array_equal(np.asarray(image.dataobj), array)
                np.testing.assert_allclose(image.affine, affine, atol=1e-5, rtol=0)
                self.assertEqual(image.shape, array.shape)
                self.assertEqual(image.header.get_xyzt_units()[0], 'mm')
                self.assertLess(image.get_fdata().min(), 0.)

    def test_resume_and_single_missing_map_repair_leave_other_maps_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, values = make_native_cache(root / 'cache')
            output = root / 'outputs/x/dti'
            export_native('x', root / 'cache', output)
            mtimes = {p: p.stat().st_mtime_ns for p in output.glob('*.nii.gz')}
            export_native('x', root / 'cache', output)
            self.assertEqual(mtimes, {p: p.stat().st_mtime_ns for p in mtimes})
            missing = output / 'x-MD.nii.gz'
            missing.unlink()
            export_native('x', root / 'cache', output)
            for p, mtime in mtimes.items():
                if p != missing:
                    self.assertEqual(p.stat().st_mtime_ns, mtime)
            np.testing.assert_array_equal(nib.load(missing).get_fdata(), values['MD'])

    def test_registered_map_cannot_be_mistaken_for_missing_native_map(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_native_cache(root / 'cache')
            (root / 'cache/x-dti-MeanDiffusivity.nhdr').rename(
                root / 'cache/x-dti-MeanDiffusivity-Reg.nhdr')
            with self.assertRaisesRegex(FileNotFoundError, 'native DTI scalar'):
                export_native('x', root / 'cache', root / 'outputs/x/dti')
            self.assertFalse((root / 'outputs').exists())

    def test_wrong_space_is_rejected_before_publishing_any_maps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_native_cache(root / 'cache')
            record = root / 'cache/raw_inputs.json'
            contents = json.loads(record.read_text())
            contents['geometry']['affine'][0][3] += 20.
            record.write_text(json.dumps(contents))
            with self.assertRaisesRegex(ValueError, 'physical space'):
                export_native('x', root / 'cache', root / 'outputs/x/dti')
            self.assertFalse(list((root / 'outputs/x/dti').glob('*.nii.gz')))

    def test_older_five_map_cache_remains_usable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_native_cache(root / 'cache', trace=False)
            with self.assertLogs(level='WARNING'):
                result = export_native('x', root / 'cache', root / 'outputs/x/dti')
            self.assertEqual(len(result['scalar_maps']), 5)
            self.assertNotIn('Trace', result['scalar_maps'])


if __name__ == '__main__':
    unittest.main()
