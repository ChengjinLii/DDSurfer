"""The compatibility check must catch changed morphometry and atlas labels."""
from pathlib import Path
import shutil
import tempfile
import unittest

import nibabel as nib
import numpy as np
import trimesh

from postprocessing.core.compare import MORPH, SURFACES, compare


class CoreComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.a, self.b = root / 'reference', root / 'candidate'
        for name in ('mri', 'surf', 'label/lh.aparc', 'stats', 'fsaverage'):
            (self.a / name).mkdir(parents=True)
        surface = trimesh.creation.icosphere(subdivisions=0)
        count = len(surface.vertices)
        geometry = dict(head=np.array([2, 0, 20], dtype=int), valid='1', filename='reference',
                        volume=np.array([16, 16, 16]), voxelsize=np.ones(3),
                        xras=np.array([-1., 0., 0.]), yras=np.array([0., 0., -1.]),
                        zras=np.array([0., 1., 0.]), cras=np.zeros(3))
        for name in SURFACES:
            nib.freesurfer.write_geometry(str(self.a / 'surf' / f'lh.{name}'), surface.vertices,
                                         surface.faces, volume_info=geometry)
        for name in MORPH:
            nib.freesurfer.write_morph_data(str(self.a / 'surf' / f'lh.{name}'), np.arange(count, dtype=np.float32))
        image = nib.MGHImage(np.zeros((count, 1, 1), np.float32), np.eye(4))
        for name in ('mri/brain.mgz', 'mri/orig.mgz'):
            nib.save(image, str(self.a / name))
        for field in ('thickness', 'curv', 'sulc', 'area.pial'):
            nib.save(image, str(self.a / 'fsaverage' / f'lh.{field}.mgh'))
        for name in ('label/lh.cortex.label', 'label/lh.aparc/lh.region.label'):
            with (self.a / name).open('w') as stream:
                stream.write(f'#!ascii label\n{count}\n')
                np.savetxt(stream, np.column_stack((np.arange(count), surface.vertices, np.zeros(count))),
                           fmt='%d %.6f %.6f %.6f %.6f')
        self.labels = np.ones(count, dtype=np.int32)
        self.colors = np.array([[0, 0, 0, 0], [100, 120, 140, 0]], dtype=np.int32)
        nib.freesurfer.write_annot(str(self.a / 'label/lh.aparc.annot'), self.labels, self.colors,
                                  [b'unknown', b'region'])
        (self.a / 'stats/lh.aparc.stats').write_text('# provenance\nregion 12 20 2.1 0.1\n')
        shutil.copytree(self.a, self.b)

    def check(self):
        return compare(self.a, self.b, ('lh',), ('aparc',))

    def test_exact_decoded_results_pass(self):
        result = self.check()
        self.assertTrue(result['all_payloads_equal'])
        self.assertEqual(result['different'], [])
        self.assertEqual(result['records']['label/lh.aparc.annot']['agreement'], 1.)

    def test_changed_thickness_and_missing_file_fail(self):
        path = self.b / 'surf/lh.thickness'
        values = nib.freesurfer.read_morph_data(str(path)).copy()
        values[0] += 1
        nib.freesurfer.write_morph_data(str(path), values)
        (self.b / 'mri/orig.mgz').unlink()
        result = self.check()
        self.assertFalse(result['all_payloads_equal'])
        self.assertIn('surf/lh.thickness', result['different'])
        self.assertIn('mri/orig.mgz', result['different'])

    def test_changed_annotation_membership_fails(self):
        self.labels[0] = 0
        nib.freesurfer.write_annot(str(self.b / 'label/lh.aparc.annot'), self.labels,
                                  self.colors, [b'unknown', b'region'])
        result = self.check()
        self.assertFalse(result['all_payloads_equal'])
        self.assertLess(result['records']['label/lh.aparc.annot']['agreement'], 1.)

    def test_stats_provenance_text_is_not_a_numerical_difference(self):
        (self.b / 'stats/lh.aparc.stats').write_text('# another date/path\nregion 12 20 2.1 0.1\n')
        result = self.check()
        self.assertTrue(result['all_payloads_equal'])
        self.assertFalse(result['records']['stats/lh.aparc.stats']['bytes_equal'])


if __name__ == '__main__':
    unittest.main()
