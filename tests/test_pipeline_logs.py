"""Concise permanent logs and disposable stage records preserve the output data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import SimpleITK as sitk
import trimesh

from inference.native import main as native_main
from preprocessing.export import main as export_main
from test_native_dti import make_native_cache
from test_raw_pipeline import make_bundle
from utils.files import atomic_json, sha256_file
from utils.surface_io import write_obj
import run_ddsurfer_pipeline as pipeline


def make_predictions(root):
    directory = root / 'mni/x'
    mesh = trimesh.creation.icosphere(subdivisions=1)
    for hemi, side in (('left', 'lh'), ('right', 'rh')):
        for kind, radius in (('wm', 1.), ('pial', 1.2)):
            source = directory / f'x_predicted_{kind}_surface_{hemi}.obj'
            write_obj(source, mesh.vertices * radius, mesh.faces)
            atomic_json(source.with_suffix('.json'), dict(
                checkpoint_file=f'ddsurfer_{side}_{kind}.pt',
                checkpoint_sha256='fixture-checkpoint', precision='fp32',
                geometry=dict(affine='detailed fixture metadata')))


class PipelineLogTests(unittest.TestCase):
    def test_command_output_is_captured_without_cluttering_the_run_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache/stage.log'
            command = [sys.executable, '-c',
                       'import sys; print("full stdout"); print("full stderr", file=sys.stderr)']
            with self.assertLogs(level='INFO') as messages:
                pipeline.run_command(command, log_path=path)
            details = path.read_text()
            self.assertIn('full stdout', details)
            self.assertIn('full stderr', details)
            self.assertTrue(details.startswith('$ '))
            summary = '\n'.join(messages.output)
            self.assertIn('Completed', summary)
            self.assertNotIn('full stdout', summary)
            self.assertNotIn('full stderr', summary)

    def test_failed_command_keeps_full_details_and_reports_only_the_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache/stage.log'
            command = [sys.executable, '-c',
                       'import sys; [print("line %02d" % i) for i in range(30)]; sys.exit(7)']
            with self.assertLogs(level='ERROR') as messages:
                with self.assertRaises(subprocess.CalledProcessError) as failure:
                    pipeline.run_command(command, log_path=path)
            self.assertEqual(failure.exception.returncode, 7)
            self.assertIn('line 00\n', path.read_text())
            summary = '\n'.join(messages.output)
            self.assertIn('line 29', summary)
            self.assertNotIn('line 00', summary)
            self.assertIn(str(path), summary)

    def test_metadata_location_does_not_change_native_surface_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'dti/x'
            data.mkdir(parents=True)
            transform = sitk.TranslationTransform(3, [2., -3., .5])
            sitk.WriteTransform(transform, str(data / 'x-b0ToAtlasT2.tfm'))
            make_predictions(root / 'predictions')
            command = ['--subject', 'x', '--data-root', str(root / 'dti'),
                       '--pred-root', str(root / 'predictions')]
            legacy = root / 'legacy/ddsurfer'
            compact = root / 'compact/ddsurfer'
            metadata = root / 'compact/.cache/logs/native'
            native_main(command + ['--output-dir', str(legacy)])
            native_main(command + ['--output-dir', str(compact), '--metadata-dir', str(metadata)])
            for source in legacy.glob('*.obj'):
                self.assertEqual(source.read_bytes(), (compact / source.name).read_bytes())
                self.assertTrue((metadata / source.with_suffix('.json').name).is_file())
            self.assertEqual(sorted(p.name for p in (compact.parent / 'logs').iterdir()),
                             ['b0ToAtlasT2.tfm'])

    def test_success_keeps_only_compact_records_and_optional_cache(self):
        for keep in (False, True):
            with self.subTest(keep=keep), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                make_bundle(root / 'inputs/x')
                argv = ['--subject', 'x', '--raw-input-root', str(root / 'inputs'),
                        '--output-root', str(root / 'outputs')]
                if keep:
                    argv += ['--keep-cache']
                args = pipeline.parse_args(argv)
                cache = pipeline.cache_directory(args)

                def run(command, log_path=None, **kwargs):
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text('stage details\n')
                    name = Path(command[1]).name
                    if name == 'run.sh':
                        make_native_cache(cache / 'dti/x')
                        sitk.WriteTransform(sitk.TranslationTransform(3, [2., -3., .5]),
                                            str(cache / 'volumes/x/x-b0ToAtlasT2.tfm'))
                    elif name == 'export.py':
                        export_main(command[2:])
                    elif name == 'DDSurfer_predict.py':
                        make_predictions(cache / 'predictions')
                    elif name == 'native.py':
                        native_main(command[2:])
                    else:
                        self.fail('Unexpected stage: ' + name)

                # Inference and tensor estimation are mocked; exports and conversion are real.
                (cache / 'volumes/x').mkdir(parents=True)
                with patch.object(pipeline, 'run_command', side_effect=run):
                    pipeline.main(argv)
                directory = pipeline.subject_directory(args)
                self.assertEqual(sorted(p.name for p in (directory / 'logs').iterdir()),
                                 ['b0ToAtlasT2.tfm', 'dti.json', 'pipeline.json',
                                  'pipeline.log', 'raw_inputs.json'])
                summary = json.loads((directory / 'logs/pipeline.json').read_text())
                self.assertTrue(summary['completed'])
                self.assertFalse(summary['postprocess'])
                self.assertEqual(len(summary['surfaces']), 4)
                for record in summary['surfaces'].values():
                    self.assertEqual(record['sha256'], sha256_file(directory / record['file']))
                    self.assertNotIn('geometry', record)
                self.assertEqual(len(list((directory / 'dti').glob('*.nii.gz'))), 6)
                self.assertEqual(cache.exists(), keep)
                if keep:
                    self.assertTrue((cache / 'state/native_dti/stages.json').is_file())
                    self.assertEqual(len(list((cache / 'logs/native').glob('*.json'))), 4)
                    self.assertTrue((cache / 'logs/preprocessing.log').is_file())

    def test_white_only_run_does_not_require_pial_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(['--subject', 'x', '--output-root', tmp,
                                        '--predict-mode', 'wm'])
            cache = pipeline.cache_directory(args)
            data = cache / 'volumes/x'
            data.mkdir(parents=True)
            sitk.WriteTransform(sitk.TranslationTransform(3, [2., -3., .5]),
                                str(data / 'x-b0ToAtlasT2.tfm'))
            make_predictions(cache / 'predictions')
            native_main(pipeline.build_native_conversion_command(args)[2:])
            pipeline.finalize_outputs(args)
            directory = pipeline.subject_directory(args)
            summary = json.loads((directory / 'logs/pipeline.json').read_text())
            self.assertEqual(set(summary['surfaces']), {'lh.white', 'rh.white'})
            self.assertEqual(sorted(p.name for p in (directory / 'ddsurfer').iterdir()),
                             ['lh.white.obj', 'rh.white.obj'])


if __name__ == '__main__':
    unittest.main()
