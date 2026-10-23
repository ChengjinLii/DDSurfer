"""Source builds remain explicit, checked and selectable alongside FreeSurfer."""
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from postprocessing.core import build, config
from postprocessing.core.archive import assets, extract
from postprocessing.core.cmake.help_header import generate
from postprocessing.core.runtime import SCHEMA, required_files, validate_runtime
from postprocessing.environment import configured_home, check_environment
from utils.files import atomic_json, sha256_file


class NativeBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / 'runtime'
        for name in required_files(('lh', 'rh'), ('aparc', 'aparc.a2009s')):
            path = self.runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture data\n')
        system, machine = config.host()
        for name in config.COMMANDS:
            path = self.runtime / 'bin' / name
            path.write_bytes((b'\x7fELF' if system == 'Linux' else b'\xcf\xfa\xed\xfe') + b'fixture')
            path.chmod(0o755)
        files = {str(path.relative_to(self.runtime)): dict(bytes=path.stat().st_size, sha256=sha256_file(path))
                 for path in self.runtime.rglob('*') if path.is_file()}
        atomic_json(self.runtime / 'runtime.json', dict(schema=SCHEMA, source_revision=config.FS_REVISION,
                    system=system, machine=machine, activation_key_included=False, files=files))
        self.license = self.root / 'private-license.txt'
        self.license.write_text('private fixture')

    def validate(self):
        return validate_runtime(self.runtime, check_libraries=False)

    def test_local_runtime_payloads_validate(self):
        self.assertEqual(self.validate()['source_revision'], config.FS_REVISION)

    def test_tampered_payload_is_not_accepted(self):
        (self.runtime / 'bin/mris_sphere').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.validate()

    def test_missing_build_has_explicit_setup_instruction(self):
        with self.assertRaisesRegex(FileNotFoundError, 'python -m postprocessing.core.build'):
            validate_runtime(self.root / 'not-built')

    def test_different_platform_is_rejected(self):
        path = self.runtime / 'runtime.json'
        data = json.loads(path.read_text()); data['machine'] = 'different'
        atomic_json(path, data)
        with self.assertRaisesRegex(ValueError, 'different platform'):
            self.validate()

    def test_private_activation_file_must_not_be_in_runtime(self):
        (self.runtime / 'license.txt').write_text('private')
        with self.assertRaisesRegex(ValueError, 'private activation'):
            self.validate()

    def test_external_symlink_is_not_accepted(self):
        path = self.runtime / 'FreeSurferColorLUT.txt'
        outside = self.root / 'outside'
        path.replace(outside); path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'external installation'):
            self.validate()

    def test_unrecorded_file_is_not_accepted(self):
        (self.runtime / 'bin/extra').write_text('unexpected')
        with self.assertRaisesRegex(ValueError, 'unrecorded'):
            self.validate()

    def test_backend_selection_preserves_both_routes(self):
        args = SimpleNamespace(postprocess_backend='auto', postprocess_runtime=None, freesurfer_home=Path('fs'))
        self.assertEqual(configured_home(args), Path('fs'))
        args.postprocess_backend = 'native'
        self.assertIsNone(configured_home(args))
        args.postprocess_runtime = self.runtime
        self.assertEqual(configured_home(args), self.runtime)
        args.postprocess_backend = 'freesurfer'
        with self.assertRaisesRegex(ValueError, 'not both'):
            configured_home(args)

    def test_native_preflight_checks_completed_build_without_compiling(self):
        with patch('postprocessing.core.config.default_prefix', return_value=self.runtime), \
                patch('postprocessing.core.runtime.shared_libraries', return_value='fixture'), \
                patch('postprocessing.core.build.build') as compile_source:
            result = check_environment(None, license_file=self.license, backend='native')
        self.assertTrue(result['minimal_runtime'])
        compile_source.assert_not_called()

    def test_existing_installation_backend_needs_a_configured_home(self):
        with self.assertRaisesRegex(ValueError, 'FREESURFER_HOME'):
            check_environment(None, backend='freesurfer')

    def test_download_cache_is_checked_even_offline(self):
        data = self.root / 'source.tar.gz'; data.write_bytes(b'cached')
        record = dict(filename=data.name, sha256=sha256_file(data), url='https://invalid.example/source')
        with patch('postprocessing.core.build.urllib.request.urlopen') as network:
            self.assertEqual(build.download(record, self.root, offline=True), data)
        network.assert_not_called()
        data.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            build.download(record, self.root, offline=True)

    def test_missing_offline_archive_does_not_access_network(self):
        record = dict(filename='missing.tar.gz', sha256='0' * 64, url='https://invalid.example/source')
        with patch('postprocessing.core.build.urllib.request.urlopen') as network:
            with self.assertRaisesRegex(FileNotFoundError, 'Offline'):
                build.download(record, self.root, offline=True)
        network.assert_not_called()

    def test_archive_traversal_and_links_are_rejected(self):
        for kind in ('traversal', 'symlink'):
            with self.subTest(kind=kind):
                archive = self.root / (kind + '.tar.gz')
                with tarfile.open(archive, 'w:gz') as stream:
                    member = tarfile.TarInfo('source/../../escaped' if kind == 'traversal' else 'source/link')
                    if kind == 'symlink':
                        member.type, member.linkname = tarfile.SYMTYPE, '/outside'
                        stream.addfile(member)
                    else:
                        member.size = 1; stream.addfile(member, io.BytesIO(b'x'))
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    extract(archive, self.root / 'unpacked', 'source')
        self.assertFalse((self.root / 'escaped').exists())

    def test_shipped_atlas_contains_no_native_executables(self):
        archive, manifest = assets()
        with tarfile.open(archive) as stream:
            names = [member.name for member in stream if member.isfile()]
        self.assertTrue(names)
        self.assertFalse(any('/bin/' in name or name.endswith(('/license.txt', '/.license')) for name in names))
        self.assertNotIn('bin/mris_sphere', manifest['files'])

    def test_help_arrays_preserve_input_bytes(self):
        source, target = self.root / 'mris_sphere.help.xml', self.root / 'help/header.h'
        source.write_bytes(b'<help>\x00\xff</help>')
        generate(source, target)
        text = target.read_text()
        self.assertIn('unsigned char mris_sphere_help_xml[]', text)
        self.assertIn('0xff', text)
        self.assertIn('mris_sphere_help_xml_len = 15;', text)

    def test_invalid_concurrency_is_rejected_before_tool_execution(self):
        with patch('postprocessing.core.build.require_tool') as tool:
            with self.assertRaisesRegex(ValueError, 'jobs'):
                build.build(SimpleNamespace(jobs=0))
        tool.assert_not_called()

    def test_named_compiler_can_be_selected_through_path(self):
        compiler = self.root / 'fakecc'; compiler.write_text('#!/bin/sh\nexit 0\n'); compiler.chmod(0o755)
        with patch.dict(os.environ, {'PATH': str(self.root)}):
            self.assertEqual(build.require_tool('fakecc', 'cc'), str(compiler))

    def test_staged_algorithm_sources_cannot_be_edited(self):
        source = self.root / 'upstream'; (source / 'include').mkdir(parents=True)
        (source / 'include/a.h').write_text('original')
        work = self.root / 'work'; work.mkdir()
        with patch('postprocessing.core.build.SOURCE_DIRECTORIES', ('include',)):
            stage = build.stage_source(source, work)
            (stage / 'include/a.h').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'edited'):
                build.stage_source(source, work)

    def test_untracked_upstream_files_are_rejected(self):
        source = self.root / 'upstream'; (source / '.git').mkdir(parents=True)
        env = dict(os.environ)
        with patch('postprocessing.core.build.subprocess.check_output', side_effect=[config.FS_REVISION, 'include/extra.h']), \
                patch('postprocessing.core.build.subprocess.run', return_value=Mock(returncode=0)):
            with self.assertRaisesRegex(ValueError, 'untracked'):
                build.prepare_source(self.root, source, 'git', True, env)

    def test_main_entry_forwards_backend_and_runtime(self):
        import run_ddsurfer_pipeline as pipeline
        args = pipeline.parse_args(['--subject', 'fixture', '--postprocess-backend', 'native',
                                    '--postprocess-runtime', str(self.runtime)])
        with patch.object(pipeline, 'native_reference', return_value=Path('b0.nii.gz')):
            command = pipeline.build_freesurfer_command(args)
        self.assertEqual(command[command.index('--backend') + 1], 'native')
        self.assertEqual(command[command.index('--runtime') + 1], str(self.runtime))


if __name__ == '__main__':
    unittest.main()
