"""Indexed mesh topology and native-coordinate regression tests."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk
import trimesh

from inference.coordinates import export_crop_mesh, Geometry
from inference.meshes import index_triangles, load_mesh, write_obj
from inference.native import atlas_ras_to_native_ras, main as native_main
from postprocess.geometry import load_pair


class MeshTests(unittest.TestCase):
    def test_indexing_does_not_round_or_merge_distinct_points(self):
        triangles = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                              [[.000001, 0., 0.], [0., 1., 0.], [1., 0., 0.]]])
        mesh, _, _ = index_triangles(triangles)
        self.assertEqual(len(mesh.vertices), 4)
        np.testing.assert_array_equal(mesh.vertices[mesh.faces], triangles)

    def test_stl_round_trip_keeps_triangle_order_and_winding(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        mesh.vertices += [10., -70., 25.]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mesh.stl'
            mesh.export(path)
            stored = load_mesh(path)
            self.assertEqual(len(stored.vertices), len(mesh.vertices))
            self.assertTrue(stored.is_watertight)
            np.testing.assert_array_equal(stored.vertices[stored.faces], mesh.vertices[mesh.faces].astype(np.float32))

    def test_obj_export_preserves_original_indices_and_coordinates(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        mesh.vertices += [10., -70., 25.]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mesh.obj'
            error = write_obj(path, mesh.vertices, mesh.faces)
            stored = load_mesh(path)
            self.assertLess(error, 1e-8)
            np.testing.assert_array_equal(stored.faces, mesh.faces)
            np.testing.assert_allclose(stored.vertices, mesh.vertices, atol=1e-8, rtol=0)

    def test_export_rejects_nonfinite_and_non_obj_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'must use .obj'):
                write_obj(Path(tmp) / 'mesh.stl', [[0., 0., 0.]], [[0, 0, 0]])
            with self.assertRaisesRegex(ValueError, 'finite'):
                write_obj(Path(tmp) / 'mesh.obj', [[np.nan, 0., 0.]], [[0, 0, 0]])
            self.assertFalse(list(Path(tmp).iterdir()))

    def test_native_conversion_outputs_only_obj_and_keeps_paired_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'dti/x'; data.mkdir(parents=True)
            transform = sitk.TranslationTransform(3, [2., -3., .5])
            sitk.WriteTransform(transform, str(data / 'x-b0ToAtlasT2.tfm'))
            affine = np.eye(4); affine[:3, 3] = [-88.299995, -129., -69.]
            for hemisphere in ('left', 'right'):
                geometry = Geometry(affine, [176, 224, 176], hemisphere)
                mesh = trimesh.creation.icosphere(subdivisions=1, radius=3.)
                for kind, radius in (('wm', 1.), ('pial', 1.2)):
                    vertices = mesh.vertices * radius + [50., 110., 88.]
                    path = root / 'pred/mni/x' / f'x_predicted_{kind}_surface_{hemisphere}.obj'
                    export_crop_mesh(path, vertices, mesh.faces, geometry)
            destination = root / 'outputs/x/ddsurfer'
            native_main(['--subject', 'x', '--data-root', str(root / 'dti'), '--pred-root', str(root / 'pred'),
                         '--output-dir', str(destination)])
            self.assertEqual(sorted(p.name for p in destination.iterdir()),
                             ['lh.pial.obj', 'lh.white.obj', 'rh.pial.obj', 'rh.white.obj'])
            for side, hemisphere in (('lh', 'left'), ('rh', 'right')):
                white, pial = load_pair(destination / f'{side}.white.obj', destination / f'{side}.pial.obj')
                np.testing.assert_array_equal(white.faces, pial.faces)
                source = load_mesh(root / 'pred/mni/x' / f'x_predicted_wm_surface_{hemisphere}.obj')
                expected = atlas_ras_to_native_ras(source.vertices[source.faces].reshape(-1, 3), transform)
                actual = white.vertices[white.faces].reshape(-1, 3)
                np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)


if __name__ == '__main__':
    unittest.main()
