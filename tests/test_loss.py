"""CPU checks for the optional one-sided boundary loss."""

from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

if find_spec('pytorch3d') is not None:
    from pytorch3d.structures import Pointclouds
    from utils.loss import boundary_loss
else:
    boundary_loss = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(boundary_loss is None, 'Optional PyTorch3D is not installed')
class BoundaryLossTests(unittest.TestCase):
    def test_direction_squared_distance_and_original_scale(self):
        x = torch.tensor([[[0., 0., 0.], [100., 0., 0.]]])
        y = torch.tensor([[[1., 0., 0.]]])
        loss, normals = boundary_loss(x, y)
        self.assertEqual(loss.item(), 2.)
        self.assertIsNone(normals)
        reverse, _ = boundary_loss(y, x)
        self.assertEqual(reverse.item(), 9802.)

    def test_loss_and_gradients_match_dense_reference(self):
        x = torch.tensor([[[0., 0., 0.], [3., 0., 0.]]], requires_grad=True)
        y = torch.tensor([[[1., 0., 0.], [4.5, 0., 0.]]], requires_grad=True)
        loss, _ = boundary_loss(x, y)
        expected = 2. * torch.cdist(y, x).square().min(dim=2).values.mean()
        torch.testing.assert_close(loss, expected)
        self.assertEqual(loss.item(), 3.25)
        actual_gradients = torch.autograd.grad(loss, (x, y))
        expected_gradients = torch.autograd.grad(expected, (x, y))
        for actual, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, reference)

    def test_padded_clouds_weights_and_reductions(self):
        x = torch.tensor([[[0., 0., 0.], [99., 0., 0.], [101., 0., 0.]],
                          [[2., 0., 0.], [5., 0., 0.], [102., 0., 0.]]])
        y = torch.tensor([[[1., 0., 0.], [4., 0., 0.], [123., 0., 0.]],
                          [[3., 0., 0.], [124., 0., 0.], [125., 0., 0.]]])
        x_lengths, y_lengths = torch.tensor([1, 2]), torch.tensor([2, 1])
        weights = torch.tensor([1., 3.])
        for point_reduction in ('mean', 'sum'):
            for batch_reduction in (None, 'mean', 'sum'):
                with self.subTest(points=point_reduction, batch=batch_reduction):
                    reduced = []
                    for index in range(2):
                        distances = torch.cdist(y[index, :y_lengths[index]],
                                                x[index, :x_lengths[index]]).square().min(dim=1).values
                        value = distances.mean() if point_reduction == 'mean' else distances.sum()
                        reduced.append(value * weights[index] * 2.)
                    expected = torch.stack(reduced)
                    if batch_reduction == 'sum':
                        expected = expected.sum()
                    elif batch_reduction == 'mean':
                        expected = expected.sum() / weights.sum()
                    loss, normals = boundary_loss(x, y, x_lengths, y_lengths, weights=weights,
                                                  point_reduction=point_reduction,
                                                  batch_reduction=batch_reduction)
                    torch.testing.assert_close(loss, expected)
                    self.assertIsNone(normals)

    def test_normals_use_absolute_cosine(self):
        x = torch.tensor([[[0., 0., 0.], [3., 0., 0.]]])
        y = torch.tensor([[[1., 0., 0.], [4.5, 0., 0.]]])
        x_normals = torch.tensor([[[1., 0., 0.], [1., 0., 0.]]])
        y_normals = torch.tensor([[[-1., 0., 0.], [0., 1., 0.]]])
        _, normal_loss = boundary_loss(x, y, x_normals=x_normals, y_normals=y_normals)
        self.assertEqual(normal_loss.item(), .5)

    def test_pointclouds_and_tensor_inputs_agree(self):
        x = [torch.tensor([[0., 0., 0.], [3., 0., 0.]]), torch.tensor([[2., 0., 0.]])]
        y = [torch.tensor([[1., 0., 0.]]), torch.tensor([[3., 0., 0.], [4., 0., 0.]])]
        a, b = Pointclouds(points=x), Pointclouds(points=y)
        loss, _ = boundary_loss(a, b)
        expected, _ = boundary_loss(a.points_padded(), b.points_padded(),
                                    a.num_points_per_cloud(), b.num_points_per_cloud())
        torch.testing.assert_close(loss, expected)

    def test_zero_weights_keep_original_zero_outputs_and_gradients(self):
        x = torch.ones((2, 3, 3), requires_grad=True)
        y = torch.zeros((2, 2, 3))
        for reduction in (None, 'mean', 'sum'):
            with self.subTest(reduction=reduction):
                loss, normals = boundary_loss(x, y, weights=torch.zeros(2), batch_reduction=reduction)
                self.assertTrue(torch.equal(loss, torch.zeros_like(loss)))
                self.assertTrue(torch.equal(normals, torch.zeros_like(normals)))
                loss.sum().backward()
                self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))
                x.grad = None

    def test_invalid_reductions_shapes_and_weights_raise(self):
        x, y = torch.zeros((1, 2, 3)), torch.zeros((1, 3, 3))
        for options in ({'point_reduction': 'max'}, {'batch_reduction': 'max'},
                        {'weights': torch.tensor([-1.])}, {'weights': torch.ones(2)},
                        {'x_lengths': torch.ones(2, dtype=torch.long)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                boundary_loss(x, y, **options)
        with self.assertRaises(ValueError):
            boundary_loss(x[0], y)


class LossDependencyTests(unittest.TestCase):
    def test_missing_pytorch3d_does_not_affect_pipeline_imports(self):
        check = '''
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class BlockPyTorch3D(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pytorch3d' or fullname.startswith('pytorch3d.'):
            raise ModuleNotFoundError('Optional PyTorch3D is unavailable', name=fullname)
sys.meta_path.insert(0, BlockPyTorch3D())
import run_ddsurfer_pipeline
import postprocessing.pipeline
assert 'utils.loss' not in sys.modules
assert 'pytorch3d' not in sys.modules
try:
    import utils.loss
except ImportError as error:
    assert 'optional training helper' in str(error), str(error)
    assert 'INSTALL.md' in str(error), str(error)
else:
    raise AssertionError('Missing PyTorch3D must be reported')
'''
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, '-I', '-B', '-c', check, str(ROOT)],
                                    cwd=temp, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
