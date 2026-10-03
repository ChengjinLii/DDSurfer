"""Regression checks for stage-independent file utilities."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from preprocessing.inputs import fingerprint
from utils.files import atomic_json, sha256_file


class FileTests(unittest.TestCase):
    def test_checksum_matches_binary_payload_across_multiple_chunks(self):
        data = b'DDSurfer\x00\xff' * 200000
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'payload.bin'
            path.write_bytes(data)
            self.assertEqual(sha256_file(path), hashlib.sha256(data).hexdigest())

    def test_atomic_json_creates_parents_and_replaces_existing_record(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'logs/record.json'
            atomic_json(path, {'completed': False})
            record = {'completed': True, 'labels': ['white', 'pial'], 'value': '\u03b1'}
            atomic_json(path, record)
            self.assertEqual(json.loads(path.read_text()), record)
            self.assertEqual(path.read_text(), json.dumps(record, indent=2, ensure_ascii=False) + '\n')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_raw_input_fingerprint_keeps_resolved_path_and_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.txt'
            source.write_bytes(b'diffusion inputs')
            link = root / 'input.txt'
            link.symlink_to(source)
            self.assertEqual(fingerprint(link), {'path': str(source.resolve()),
                             'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})


if __name__ == '__main__':
    unittest.main()
