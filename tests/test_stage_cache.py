"""Failure, corruption and dependency checks for resumable pipeline stages."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from preprocessing.cache import related_files, validate_output
from utils.stages import StageCache, subject_lock


class StageCacheTests(unittest.TestCase):
    def test_reuse_corruption_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'source', root / 'result'
            source.write_text('original')
            cache = StageCache(root / 'stages.json')
            calls = []
            def action(staged):
                calls.append(True)
                staged[0].write_text(source.read_text())
            self.assertTrue(cache.run('copy', [source], [output], action, settings={'shape': (1, 2)}))
            cache = StageCache(root / 'stages.json')
            self.assertFalse(cache.run('copy', [source], [output], action, settings={'shape': (1, 2)}))
            output.write_text('damaged')
            self.assertTrue(cache.run('copy', [source], [output], action, settings={'shape': (1, 2)}))
            source.write_text('new source')
            self.assertTrue(cache.run('copy', [source], [output], action, settings={'shape': (1, 2)}))
            self.assertEqual(output.read_text(), 'new source')
            self.assertEqual(len(calls), 3)

    def test_failure_does_not_publish_partial_files_or_leave_a_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, first, second = (root / name for name in ('source', 'first', 'second'))
            source.write_text('input')
            cache = StageCache(root / 'stages.json')
            def complete(staged):
                for path in staged:
                    path.write_text('complete')
            cache.run('pair', [source], [first, second], complete)
            source.write_text('changed')
            def fail(staged):
                staged[0].write_text('partial')
                raise RuntimeError('interrupted')
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                cache.run('pair', [source], [first, second], fail)
            self.assertEqual(first.read_text(), 'complete')
            self.assertEqual(second.read_text(), 'complete')
            self.assertNotIn('pair', json.loads(cache.path.read_text()))
            self.assertFalse(list(root.glob('.stage-*')))
            self.assertTrue(cache.run('pair', [source], [first, second], complete))

    def test_detached_nrrd_payload_is_published_and_missing_payload_is_repaired(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / 'tensor.nhdr'
            cache = StageCache(root / 'stages.json', related_files, validate_output)
            image = sitk.GetImageFromArray(np.ones((3, 4, 5), np.float32))
            action = lambda staged: sitk.WriteImage(image, str(staged[0]), True)
            self.assertTrue(cache.run('tensor', [], [output], action))
            self.assertFalse(cache.run('tensor', [], [output], action))
            related_files(output)[1].unlink()
            self.assertTrue(cache.run('tensor', [], [output], action))
            np.testing.assert_array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(output))), 1.)
            output.write_text('invalid header')
            self.assertTrue(cache.run('tensor', [], [output], action))

    def test_empty_output_is_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / 'result'
            cache = StageCache(root / 'stages.json')
            with self.assertRaisesRegex(ValueError, 'empty'):
                cache.run('empty', [], [output], lambda staged: staged[0].touch())
            self.assertFalse(output.exists())
            self.assertNotIn('empty', cache.records)

    def test_truncated_detached_data_is_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / 'tensor.nhdr'
            cache = StageCache(root / 'stages.json', related_files, validate_output)
            def truncated(staged):
                sitk.WriteImage(sitk.GetImageFromArray(np.ones((3,4,5), np.float32)), str(staged[0]), True)
                payload = related_files(staged[0])[1]
                payload.write_bytes(payload.read_bytes()[:10])
            with self.assertRaises((EOFError, ValueError, OSError)):
                cache.run('tensor', [], [output], truncated)
            self.assertFalse(output.exists())
            self.assertNotIn('tensor', cache.records)

    def test_subject_cache_cannot_be_used_by_two_writers(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'lock'
            with subject_lock(path):
                with self.assertRaisesRegex(RuntimeError, 'Another process'):
                    with subject_lock(path):
                        self.fail('Lock unexpectedly acquired')
            with subject_lock(path):
                pass

    def test_absolute_and_relative_input_paths_reuse_the_same_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'source', root / 'result'
            source.write_text('input')
            cache = StageCache(root / 'stages.json')
            action = lambda staged: staged[0].write_text('complete')
            self.assertTrue(cache.run('copy', [source], [output], action))
            relative = Path(os.path.relpath(source))
            self.assertFalse(cache.run('copy', [relative], [output], action))


if __name__ == '__main__':
    unittest.main()
