"""Geometry checks for native-space FreeSurfer postprocessing."""
from pathlib import Path
import json
import os
import shutil
import tempfile
from threading import Lock
import time
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import trimesh

import run_ddsurfer_pipeline as pipeline
from postprocess.geometry import load_pair, load_volume, prepare_brain, prepare_surface_reference, vertex_area, volume_files, write_surface
from postprocess.pipeline import parse_args, run as run_postprocess
from postprocess.stats import write_stats


class PostprocessTests(unittest.TestCase):
    def test_reference_reorientation_preserves_physical_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            affine = np.array([[0., -1.25, 0., 24.], [1.5, 0., 0., -19.], [0., 0., 2., -10.], [0., 0., 0., 1.]])
            data = np.arange(20 * 24 * 18, dtype=np.float32).reshape(20, 24, 18) + 1.
            source = root / 'native.nii.gz'
            nib.save(nib.Nifti1Image(data, affine), source)
            brain = prepare_brain(source, root / 'brain.mgz')
            self.assertEqual(nib.aff2axcodes(brain.affine), ('L', 'I', 'A'))
            scaled = np.clip(data / np.percentile(data, 99.5) * 255., 0, 255).astype(np.uint8)
            source_img = nib.Nifti1Image(scaled, affine)
            orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(affine),
                                                         nib.orientations.axcodes2ornt('LIA'))
            expected_img = source_img.as_reoriented(orientation)
            np.testing.assert_allclose(brain.affine, expected_img.affine, atol=1e-4)
            np.testing.assert_array_equal(np.asarray(brain.dataobj), expected_img.dataobj)
            np.testing.assert_allclose(brain.header.get_vox2ras_tkr()[:3, :3], brain.affine[:3, :3], atol=1e-5)
            mesh = trimesh.creation.icosphere(subdivisions=1, radius=3.)
            mesh.vertices += nib.affines.apply_affine(affine, np.array(data.shape) / 2.)
            qc = write_surface(mesh, brain, root / 'lh.white')
            self.assertLess(qc['scanner_RAS_roundtrip_max_error_mm'], 1e-4)
            self.assertEqual(qc['fraction_inside_reference'], 1.)
            vertices, faces, info = nib.freesurfer.read_geometry(root / 'lh.white', read_metadata=True)
            self.assertFalse(qc['triangle_winding_reversed'])
            np.testing.assert_array_equal(faces, mesh.faces)
            self.assertGreater(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).volume, 0.)
            axes = np.column_stack([info['xras'], info['yras'], info['zras']])
            stored_affine = np.eye(4)
            stored_affine[:3, :3] = axes @ np.diag(info['voxelsize'])
            stored_affine[:3, 3] = info['cras'] - stored_affine[:3, :3] @ (info['volume'] / 2.)
            np.testing.assert_allclose(stored_affine, brain.affine, atol=1e-4)
            expected = nib.affines.apply_affine(stored_affine @ np.linalg.inv(brain.header.get_vox2ras_tkr()), vertices)
            np.testing.assert_allclose(expected, mesh.vertices, atol=1e-4)
            np.testing.assert_allclose(np.linalg.norm(info['xras']), 1.)

    def test_las_reference_cannot_rotate_registration_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            affine = np.diag([-1.25, 1.25, 1.25, 1.])
            affine[:3, 3] = [90., -126., -72.]
            source = root / 'b0.nii.gz'
            nib.save(nib.Nifti1Image(np.ones((145, 174, 145), np.float32), affine), source)
            brain = prepare_brain(source, root / 'brain.mgz')
            scanner_to_tkr = brain.header.get_vox2ras_tkr() @ np.linalg.inv(brain.affine)
            np.testing.assert_allclose(scanner_to_tkr[:3, :3], np.eye(3), atol=1e-5)
            self.assertEqual(brain.shape, (145, 145, 174))

    def test_oblique_reference_reslice_keeps_native_landmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            angle = .2
            affine = np.array([[np.cos(angle), -np.sin(angle), 0., -20.],
                               [np.sin(angle), np.cos(angle), 0., -20.],
                               [0., 0., 1.5, -22.], [0., 0., 0., 1.]])
            indices = np.indices((32, 32, 32)).astype(float)
            data = np.exp(-np.sum((indices - 16.) ** 2, axis=0) / 3.).astype(np.float32)
            source = root / 'native.nii.gz'
            nib.save(nib.Nifti1Image(data, affine), source)
            brain = prepare_brain(source, root / 'brain.mgz')
            np.testing.assert_allclose(brain.header.get_vox2ras_tkr()[:3, :3], brain.affine[:3, :3], atol=1e-5)
            values = np.asarray(brain.dataobj, dtype=float)
            grid = np.indices(brain.shape, dtype=float)
            maximum = (grid * values).reshape(3, -1).sum(axis=1) / values.sum()
            expected = nib.affines.apply_affine(affine, [16., 16., 16.])
            actual = nib.affines.apply_affine(brain.affine, maximum)
            self.assertLess(float(np.linalg.norm(expected - actual)), 1.5)

    def test_nhdr_lps_to_ras_matches_nifti(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = sitk.GetImageFromArray(np.ones((9, 10, 11), dtype=np.float32))
            image.SetOrigin((10., 20., -8.)); image.SetSpacing((1.25, 1.5, 2.))
            sitk.WriteImage(image, str(root / 'b0.nhdr'))
            sitk.WriteImage(image, str(root / 'b0.nii.gz'))
            a, affine_a = load_volume(root / 'b0.nhdr')
            files = volume_files(root / 'b0.nhdr')
            self.assertEqual(len(files), 2)
            self.assertTrue(files[1].is_file())
            b, affine_b = load_volume(root / 'b0.nii.gz')
            np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(affine_a, affine_b)

    def test_reject_surface_reference_space_mismatch(self):
        image = nib.MGHImage(np.ones((10, 10, 10), dtype=np.uint8), np.eye(4))
        mesh = trimesh.creation.icosphere(subdivisions=1)
        mesh.vertices += 100.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'outside'):
                write_surface(mesh, image, Path(tmp) / 'lh.white')

    def test_reject_white_pial_index_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh = trimesh.creation.icosphere(subdivisions=1)
            mesh.export(root / 'wm.obj')
            mesh.faces = mesh.faces[::-1]
            mesh.export(root / 'pial.obj')
            with self.assertRaisesRegex(ValueError, 'matching vertex indices'):
                load_pair(root / 'wm.obj', root / 'pial.obj')

    def test_area_is_conserved(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        self.assertAlmostEqual(float(vertex_area(mesh.vertices, mesh.faces).sum()), mesh.area, places=5)

    def test_native_reference_not_taken_from_mni_preprocessed_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(['--subject', 'x', '--output-root', tmp])
            directory = pipeline.cache_directory(args) / 'dti/x'
            directory.mkdir(parents=True)
            source = directory / 'x-b0.nhdr'
            source.write_text('fixture')
            self.assertEqual(pipeline.native_reference(args), source)
            command = pipeline.build_freesurfer_command(args)
            self.assertEqual(command[command.index('--brain-source') + 1], str(source))
            for flag in ('--lh-white', '--lh-pial', '--rh-white', '--rh-pial'):
                file = Path(command[command.index(flag) + 1])
                self.assertEqual(file.suffix, '.obj')
                self.assertEqual(file.parent, pipeline.subject_directory(args) / 'ddsurfer')

    def test_pipeline_stage_order_and_shell_aliases(self):
        raw = {name: Path('raw') / name for name in ('dwi', 'bval', 'bvec', 'mask')}
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(pipeline, 'run_command') as run, patch.object(pipeline, 'resolve_inputs', return_value=raw), \
                    patch.object(pipeline, 'finalize_outputs'):
                pipeline.main(['--subject', 'x', '--post-process', '--freesurfer-home', './freesurfer',
                               '--brain-source', './native-b0.nii.gz', '--output-root', temp])
        names = [Path(call[0][0][1]).name for call in run.call_args_list]
        self.assertEqual(names, ['run.sh', 'DDSurfer_predict.py', 'native.py', 'pipeline.py'])

    def test_freesurfer_requires_pial_before_starting(self):
        with patch.object(pipeline, 'run_command') as run:
            with self.assertRaisesRegex(ValueError, 'white and pial'):
                pipeline.main(['--subject', 'x', '--freesurfer', '--predict-mode', 'wm'])
        run.assert_not_called()

    def test_bilateral_parallel_run_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / 'fs'
            (home / 'bin').mkdir(parents=True)
            (home / 'average').mkdir()
            labels = home / 'subjects/fsaverage/label'
            labels.mkdir(parents=True)
            (home / 'license.txt').write_text('fixture')
            for binary in ('mris_curvature', 'mris_inflate', 'mris_sphere', 'mris_register', 'mris_thickness',
                           'mri_label2label', 'mri_surf2surf', 'mri_annotation2label', 'mris_anatomical_stats',
                           'mris_convert', 'mris_place_surface'):
                path = home / 'bin' / binary
                path.write_text('#!/bin/sh\nexit 0\n'); path.chmod(0o755)
            native = root / 'native/x'
            native.mkdir(parents=True)
            vertices_count = 42
            for hemi in ('lh', 'rh'):
                (labels / f'{hemi}.cortex.label').write_text('fixture')
                (home / 'average' / f'{hemi}.average.curvature.filled.buckner40.tif').write_text('fixture')
                nib.freesurfer.write_annot(str(labels / f'{hemi}.aparc.annot'),
                                          np.ones(vertices_count, dtype=int),
                                          np.array([[0, 0, 0, 0], [10, 20, 30, 0], [40, 50, 60, 0]]),
                                          [b'unknown', b'roi', b'corpuscallosum'])
                for kind, radius in (('wm', 2.), ('pial', 3.)):
                    mesh = trimesh.creation.icosphere(subdivisions=1, radius=radius)
                    mesh.vertices += 10.
                    mesh.export(native / f'x_predicted_{kind}_{hemi}.obj')
            brain = root / 'b0.nii.gz'
            nib.save(nib.Nifti1Image(np.ones((20, 20, 20), np.float32), np.eye(4)), brain)
            args = parse_args(['--subject', 'x', '--native-root', str(root / 'native'), '--brain-source', str(brain),
                               '--output-root', str(root / 'output'), '--freesurfer-home', str(home), '--atlases', 'aparc'])
            lock = Lock()
            activity = [0, 0]

            def mock_command(command, **kwargs):
                with lock:
                    activity[0] += 1
                    activity[1] = max(activity)
                time.sleep(.01)
                self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '2')
                tool = command[0]
                if tool == 'mris_curvature':
                    for suffix in ('.H', '.K'):
                        nib.freesurfer.write_morph_data(command[-1] + suffix, np.ones(vertices_count))
                elif tool == 'mris_place_surface':
                    nib.freesurfer.write_morph_data(command[-1], np.ones(vertices_count))
                elif tool in ('mris_inflate', 'mris_sphere', 'mris_register'):
                    shutil.copyfile(command[-2] if tool != 'mris_register' else command[-3], command[-1])
                    if tool == 'mris_inflate':
                        nib.freesurfer.write_morph_data(command[-1].replace('.inflated', '.sulc'), np.ones(vertices_count))
                elif tool == 'mri_label2label':
                    Path(command[command.index('--trglabel') + 1]).write_text(
                        '# cortex\n42\n' + ''.join(f'{i} 0 0 0 0\n' for i in range(vertices_count)))
                elif tool == 'mri_surf2surf':
                    out = Path(command[command.index('--tval') + 1])
                    if '--sval-annot' in command:
                        shutil.copyfile(command[command.index('--sval-annot') + 1], out)
                    else:
                        out.write_text('overlay fixture')
                elif tool == 'mri_annotation2label':
                    hemi = command[command.index('--hemi') + 1]
                    (Path(command[-1]) / f'{hemi}.roi.label').write_text('fixture')
                elif Path(command[1]).name == 'stats.py':
                    rows = write_stats(command[command.index('--subject-dir') + 1],
                                       command[command.index('--hemi') + 1], command[-1])
                    self.assertEqual(rows[0][0], 'roi')
                    self.assertEqual(rows[0][1], 42)
                    self.assertEqual(rows[0][2], 42.)
                    self.assertEqual(rows[0][4], 1.)
                else:
                    self.fail(tool)
                with lock:
                    activity[0] -= 1

            with patch.dict(os.environ, FS_LICENSE=str(home / 'license.txt')):
                with patch('postprocess.pipeline.subprocess.run', side_effect=mock_command) as process:
                    run_postprocess(args)
                    self.assertEqual(activity[1], 2)
                    self.assertGreater(process.call_count, 20)
                    process.reset_mock()
                    run_postprocess(args)
                    process.assert_not_called()
                    # Missing outputs must be regenerated, not treated as completed.
                    (root / 'output/x/surf/lh.thickness').unlink()
                    run_postprocess(args)
                    self.assertEqual(process.call_count, 1)
                    explicit = parse_args(['--subject', 'x', '--output-root', str(root / 'surface-only'),
                                           '--freesurfer-home', str(home), '--atlases', 'aparc'] +
                                          [value for hemi in ('lh', 'rh') for kind, source in (('white', 'wm'), ('pial', 'pial'))
                                           for value in (f'--{hemi}-{kind}', str(native / f'x_predicted_{source}_{hemi}.obj'))])
                    run_postprocess(explicit)
                    reference = nib.load(root / 'surface-only/x/mri/brain.mgz')
                    self.assertEqual(nib.aff2axcodes(reference.affine), ('L', 'I', 'A'))
                    self.assertEqual(np.count_nonzero(np.asarray(reference.dataobj)), 0)
                    check = json.loads((root / 'surface-only/x/logs/qc.json').read_text())
                    self.assertEqual(check['reference_content'], 'geometry_only')
            qc = json.loads((root / 'output/x/logs/qc.json').read_text())
            self.assertEqual(set(qc['hemispheres']), {'lh', 'rh'})
            signature_path = root / 'output/x/logs/inputs.json'
            signature = json.loads(signature_path.read_text())
            self.assertEqual(signature.pop('reference_orientation'), 'LIA')
            signature_path.write_text(json.dumps(signature))
            with self.assertRaisesRegex(ValueError, 'different inputs/settings'):
                run_postprocess(args)

    def test_four_surfaces_reference_preserves_native_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meshes = [trimesh.creation.icosphere(subdivisions=1, radius=radius) for radius in (5., 7.)]
            for mesh in meshes:
                mesh.vertices += [-20., 30., -40.]
            image = prepare_surface_reference(meshes, root / 'brain.mgz')
            np.testing.assert_allclose((image.header.get_vox2ras_tkr() @ np.linalg.inv(image.affine))[:3, :3], np.eye(3))
            for kind, mesh in zip(('white', 'pial'), meshes):
                write_surface(mesh, image, root / f'lh.{kind}')
            white, pial = load_pair(root / 'lh.white', root / 'lh.pial')
            for actual, expected in zip((white, pial), meshes):
                np.testing.assert_allclose(actual.vertices, expected.vertices, atol=1e-4)
                np.testing.assert_array_equal(actual.faces, expected.faces)

    def test_stl_welding_uses_shared_white_pial_vertex_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            white = trimesh.creation.icosphere(subdivisions=1, radius=3.)
            pial = white.copy(); pial.vertices *= 1.2
            white.export(root / 'white.stl'); pial.export(root / 'pial.stl')
            w, p = load_pair(root / 'white.stl', root / 'pial.stl')
            self.assertEqual(len(w.vertices), len(white.vertices))
            self.assertTrue(w.is_watertight)
            np.testing.assert_allclose(p.vertices, w.vertices * 1.2, atol=1e-5)
            np.testing.assert_array_equal(w.faces, p.faces)
            pial.faces = np.roll(pial.faces, 1, axis=0)
            pial.export(root / 'pial.stl')
            with self.assertRaisesRegex(ValueError, 'correspondence'):
                load_pair(root / 'white.stl', root / 'pial.stl')

    def test_indexed_postprocess_formats_preserve_pair_correspondence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            white = trimesh.creation.icosphere(subdivisions=1)
            pial = white.copy(); pial.vertices *= 1.2
            for suffix in ('obj', 'ply', 'off'):
                with self.subTest(suffix=suffix):
                    white.export(root / f'white.{suffix}'); pial.export(root / f'pial.{suffix}')
                    actual_white, actual_pial = load_pair(root / f'white.{suffix}', root / f'pial.{suffix}')
                    self.assertEqual(len(actual_white.vertices), len(white.vertices))
                    np.testing.assert_array_equal(actual_white.faces, white.faces)
                    np.testing.assert_array_equal(actual_white.faces, actual_pial.faces)
                    np.testing.assert_allclose(actual_pial.vertices, pial.vertices, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
