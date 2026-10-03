"""Checks for model components and shared processing helpers."""
import ast
from pathlib import Path
import unittest

import numpy as np
import torch

from model import cmunext
from model.attention import DeformableConv2D, LKAAttention3D, LargeKernelAttention3D
from model.cmunext import (CMUNeXtBlock3D, CMUNeXtSEBlock3D, ConvBlock3D,
                           FusionBlock3D, SEFusionBlock3D, UpsampleBlock3D)
from model.ddsurfer import CrossStreamFusion, TANet, TemporalAttentionNet, VelocityFieldNet
from model.volume import SeparableConv3D, SeparableUpConv3D, VolumeAttention3D
from preprocessing.conversion import encode_nrrd_gradient, write_nhdr, write_nifti
from preprocessing.conversion.bval_bvec_io import transpose
from preprocessing.resample import _parse_direction

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_model_class_names_use_consistent_case(self):
        for path in (ROOT / 'model').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ClassDef):
                    with self.subTest(file=path.name, name=node.name):
                        self.assertRegex(node.name, r'^[A-Z][A-Za-z0-9]*$')

    def test_cmunext_exports_are_resolvable(self):
        self.assertEqual(len(cmunext.__all__), len(set(cmunext.__all__)))
        for name in cmunext.__all__:
            self.assertTrue(issubclass(getattr(cmunext, name), torch.nn.Module))

    def test_surface_network_has_descriptive_components(self):
        for refinement in (None, {}):
            with self.subTest(refinement=refinement):
                network = TANet(C_hid=(8,16,24,32,48), inshape=(16,32,16),
                                depths=(1,1,1,1,1), kernels=(3,3,3,3,3),
                                step_size=0.5, refinement=refinement).eval()
                self.assertIsInstance(network.vf_net, VelocityFieldNet)
                self.assertIsInstance(network.att_net, TemporalAttentionNet)
                self.assertIsInstance(network.vf_net.fusion5, CrossStreamFusion)
                vertices = torch.tensor([[[4.,8.,4.], [8.,16.,8.], [12.,24.,12.]]])
                with torch.no_grad():
                    result, extras = network(vertices, torch.randn(1,5,16,32,16),
                                             return_extras=True)
                self.assertEqual(result.shape, vertices.shape)
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(extras['svfs_all'].shape, (6,3,16,32,16))

    def test_3d_blocks_preserve_expected_shapes(self):
        cases = ((ConvBlock3D(8,12), 8, (4,4,4), 12, (4,4,4)),
                 (CMUNeXtBlock3D(8,12), 8, (4,4,4), 12, (4,4,4)),
                 (CMUNeXtSEBlock3D(8,12), 8, (4,4,4), 12, (4,4,4)),
                 (UpsampleBlock3D(8,12), 8, (4,4,4), 12, (8,8,8)),
                 (FusionBlock3D(16,8), 16, (4,4,4), 8, (4,4,4)),
                 (SEFusionBlock3D(16,8), 16, (4,4,4), 8, (4,4,4)),
                 (SeparableConv3D(8,12,1), 8, (4,4,4), 12, (4,4,4)),
                 (SeparableUpConv3D(8,12,2), 8, (4,4,4), 12, (8,8,8)))
        for block, channels, shape, out_channels, out_shape in cases:
            with self.subTest(block=type(block).__name__):
                with torch.no_grad():
                    result = block.eval()(torch.randn(1, channels, *shape))
                self.assertEqual(result.shape, (1, out_channels, *out_shape))
                self.assertTrue(torch.isfinite(result).all())

    def test_attention_blocks_preserve_3d_shape(self):
        features = torch.randn(1,16,4,6,4)
        for block in (LKAAttention3D(16), LargeKernelAttention3D(16),
                      VolumeAttention3D(16, reduction_ratio=4)):
            with self.subTest(block=type(block).__name__):
                with torch.no_grad():
                    result = block.eval()(features)
                self.assertEqual(result.shape, features.shape)
                self.assertTrue(torch.isfinite(result).all())

    def test_deformable_2d_block_matches_its_actual_dimensions(self):
        block = DeformableConv2D(4, groups=2).eval()
        features = torch.randn(1,4,8,8)
        self.assertIsInstance(block.offset_net, torch.nn.Conv2d)
        with torch.no_grad():
            result = block(features)
        self.assertEqual(result.shape, features.shape)

    def test_conversion_helpers_have_clear_exports(self):
        self.assertTrue(callable(write_nhdr))
        self.assertTrue(callable(write_nifti))
        self.assertEqual(transpose([[1,2,3],[4,5,6]]), [[1,4],[2,5],[3,6]])
        self.assertEqual(transpose(transpose([[1,2,3],[4,5,6]])), [[1,2,3],[4,5,6]])
        encoded = encode_nrrd_gradient(250., [1.,0.,0.], 1000.)
        np.testing.assert_array_equal(np.fromstring(encoded, sep=' '), [.5,0.,0.])

    def test_direction_parser_preserves_values(self):
        direction = (1,0,0,0,1,0,0,0,1)
        self.assertEqual(_parse_direction(direction), tuple(float(value) for value in direction))
        with self.assertRaises(ValueError):
            _parse_direction(direction[:-1])


if __name__ == '__main__':
    unittest.main()
