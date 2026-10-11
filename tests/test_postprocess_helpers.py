"""CPU regression checks for label export and batched surface overlays."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from postprocessing.labels import read_regions, write_labels
from postprocessing.overlays import FIELDS, copy_frames, write_stack
from postprocessing.stats import read_measurements, write_stats


def annotation(path, names=(b'unknown', b'roi', b'absent')):
    table = np.array([[25, 5, 25, 0], [10, 20, 30, 0], [40, 50, 60, 0]])
    nib.freesurfer.write_annot(str(path), np.array([1, 0, 1, 1]), table, list(names))


class PostprocessHelperTests(unittest.TestCase):
    def test_labels_preserve_colored_unknown_and_vertex_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation(root / 'aparc.annot')
            count, regions = read_regions(root / 'aparc.annot')
            self.assertEqual(count, 4)
            self.assertEqual([name for name, _ in regions], ['unknown', 'roi'])
            np.testing.assert_array_equal(regions[0][1], [1])
            np.testing.assert_array_equal(regions[1][1], [0, 2, 3])
            vertices = np.array([[1.125, -2.5, 3.75], [4., 5., 6.], [7., 8., 9.], [-0., 0., 0.]])
            nib.freesurfer.write_geometry(str(root / 'lh.white'), vertices, np.array([[0, 1, 2], [0, 2, 3]]))
            outputs = [root / f'lh.{name}.label' for name, _ in regions]
            write_labels(root / 'lh.white', 'x', count, regions, outputs)
            self.assertEqual(outputs[1].read_text(),
                             '#!ascii label  , from subject x vox2ras=TkReg\n3\n'
                             '0  1.125  -2.500  3.750 0.0000000000\n'
                             '2  7.000  8.000  9.000 0.0000000000\n'
                             '3  -0.000  0.000  0.000 0.0000000000\n')

    def test_black_unknown_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'a.annot'
            nib.freesurfer.write_annot(str(path), np.array([0, 1]),
                                      np.array([[0, 0, 0, 0], [1, 2, 3, 0]]), [b'unknown', b'roi'])
            _, regions = read_regions(path)
            self.assertEqual([name for name, _ in regions], ['unknown', 'roi'])

    def test_unsafe_region_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'a.annot'
            annotation(path, (b'unknown', b'../escape', b'absent'))
            with self.assertRaisesRegex(ValueError, 'region name'):
                read_regions(path)

    def test_duplicate_region_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'a.annot'
            annotation(path, (b'roi', b'roi', b'absent'))
            with self.assertRaisesRegex(ValueError, 'unique'):
                read_regions(path)

    def test_label_count_mismatch_does_not_publish_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nib.freesurfer.write_geometry(str(root / 'lh.white'), np.zeros((3, 3)), np.array([[0, 1, 2]]))
            out = root / 'roi.label'
            with self.assertRaisesRegex(ValueError, 'vertices disagree'):
                write_labels(root / 'lh.white', 'x', 4, [('roi', np.array([0]))], [out])
            self.assertFalse(out.exists())

    def test_overlay_stack_preserves_values_and_legacy_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for frame, field in enumerate(FIELDS):
                nib.freesurfer.write_morph_data(str(root / f'lh.{field}'), np.arange(7, dtype=np.float32) + frame)
            write_stack(root, 'lh', root / 'stack.mgh', 7)
            image = nib.MGHImage.from_bytes((root / 'stack.mgh').read_bytes())
            self.assertEqual(image.shape, (7, 1, 1, 4))
            self.assertEqual(int(image.header['dof']), 1)
            np.testing.assert_array_equal(image.header['Pxyz_c'], [0, 0, 0])
            np.testing.assert_array_equal(image.header['Mdc'], [[-1, 0, 0], [0, 0, -1], [0, 1, 0]])
            outputs = [root / f'{field}.mgh' for field in FIELDS]
            for frame in range(len(FIELDS)):
                path = root / f'mapped{frame:04d}.mgh'
                nib.save(nib.MGHImage(np.asarray(image.dataobj)[..., frame], image.affine,
                                     image.header.copy()), str(path))
                with path.open('ab') as stream:
                    stream.write(b'opaque FreeSurfer metadata fixture')
            copy_frames(root / 'mapped.mgh', outputs)
            for frame, output in enumerate(outputs):
                result = nib.MGHImage.from_bytes(output.read_bytes())
                self.assertEqual(result.shape, (7, 1, 1))
                np.testing.assert_array_equal(result.affine, image.affine)
                np.testing.assert_array_equal(np.asarray(result.dataobj).ravel(), np.arange(7) + frame)
                self.assertEqual(output.read_bytes(), (root / f'mapped{frame:04d}.mgh').read_bytes())

    def test_invalid_source_overlay_does_not_create_stack(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for field in FIELDS:
                nib.freesurfer.write_morph_data(str(root / f'lh.{field}'), np.ones(7))
            for values in (np.ones(6), np.array([np.nan] + [1.] * 6)):
                nib.freesurfer.write_morph_data(str(root / 'lh.curv'), values)
                with self.assertRaisesRegex(ValueError, 'overlay'):
                    write_stack(root, 'lh', root / 'stack.mgh', 7)
                self.assertFalse((root / 'stack.mgh').exists())

    def test_invalid_mapped_frame_does_not_overwrite_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = [root / f'{field}.mgh' for field in FIELDS]
            for path in outputs:
                path.write_bytes(b'previous output')
            for frame in range(len(FIELDS)):
                nib.save(nib.MGHImage(np.ones((7, 1, 1), np.float32), np.eye(4)),
                         str(root / f'mapped{frame:04d}.mgh'))
            cases = [np.ones((7, 1, 1, 3), np.float32), np.full((7, 1, 1), np.nan, np.float32),
                     np.ones((6, 1, 1), np.float32)]
            for data in cases:
                nib.save(nib.MGHImage(data, np.eye(4)), str(root / 'mapped0003.mgh'))
                with self.assertRaisesRegex(ValueError, '[Mm]apped surface'):
                    copy_frames(root / 'mapped.mgh', outputs)
                self.assertTrue(all(path.read_bytes() == b'previous output' for path in outputs))

    def test_missing_mapped_frame_does_not_overwrite_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = [root / f'{field}.mgh' for field in FIELDS]
            for path in outputs:
                path.write_bytes(b'previous output')
            for frame in range(len(FIELDS) - 1):
                nib.save(nib.MGHImage(np.ones((7, 1, 1), np.float32), np.eye(4)),
                         str(root / f'mapped{frame:04d}.mgh'))
            with self.assertRaises(FileNotFoundError):
                copy_frames(root / 'mapped.mgh', outputs)
            self.assertTrue(all(path.read_bytes() == b'previous output' for path in outputs))

    def test_staged_stats_match_original_writer_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'x'
            (root / 'surf').mkdir(parents=True)
            (root / 'label').mkdir()
            annotation(root / 'label/lh.aparc.annot')
            (root / 'label/lh.cortex.label').write_text('# cortex\n4\n' + ''.join(f'{i} 0 0 0 0\n' for i in range(4)))
            for field in ('area', 'area.pial', 'thickness', 'curv', 'white.K'):
                nib.freesurfer.write_morph_data(str(root / 'surf' / f'lh.{field}'), np.array([1., 2., 3., 4.]))
            rows = write_stats(root, 'lh', 'aparc')
            staged = root / 'staged/stats.txt'
            self.assertEqual(rows, write_stats(root, 'lh', 'aparc', output=staged))
            self.assertEqual(staged.read_bytes(), (root / 'stats/lh.aparc.stats').read_bytes())
            measurements = read_measurements(root, 'lh')
            with patch('nibabel.freesurfer.read_label') as read_label, \
                    patch('nibabel.freesurfer.read_morph_data') as read_maps:
                self.assertEqual(rows, write_stats(root, 'lh', 'aparc', output=staged, measurements=measurements))
                read_label.assert_not_called()
                read_maps.assert_not_called()
            self.assertEqual(staged.read_bytes(), (root / 'stats/lh.aparc.stats').read_bytes())

    def test_cached_label_vertices_preserve_bytes_without_reading_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vertices = np.array([[1.125, -2.5, 3.75], [4., 5., 6.], [7., 8., 9.]])
            nib.freesurfer.write_geometry(str(root / 'lh.white'), vertices, np.array([[0, 1, 2]]))
            regions = [('roi', np.arange(3))]
            write_labels(root / 'lh.white', 'x', 3, regions, [root / 'original.label'])
            with patch('nibabel.freesurfer.read_geometry') as read:
                write_labels(root / 'lh.white', 'x', 3, regions, [root / 'cached.label'], vertices=vertices)
                read.assert_not_called()
            self.assertEqual((root / 'original.label').read_bytes(), (root / 'cached.label').read_bytes())

    def test_cached_overlay_values_preserve_stack_bytes_and_validate_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fields = {}
            for frame, field in enumerate(FIELDS):
                path = root / f'lh.{field}'
                nib.freesurfer.write_morph_data(str(path), np.arange(7, dtype=np.float32) + frame)
                fields[field] = nib.freesurfer.read_morph_data(str(path))
            write_stack(root, 'lh', root / 'original.mgh', 7)
            with patch('nibabel.freesurfer.read_morph_data') as read:
                write_stack(root, 'lh', root / 'cached.mgh', 7, fields=fields)
                read.assert_not_called()
            self.assertEqual((root / 'original.mgh').read_bytes(), (root / 'cached.mgh').read_bytes())
            fields['curv'] = np.full(7, np.nan, np.float32)
            with self.assertRaisesRegex(ValueError, 'overlay'):
                write_stack(root, 'lh', root / 'bad.mgh', 7, fields=fields)
            self.assertFalse((root / 'bad.mgh').exists())


if __name__ == '__main__':
    unittest.main()
