"""Deformable large-kernel attention primitives for DDSurfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = ["batch_map_offsets", "ConvOffset3D", "deform_conv3d", "LKA_Attention3d"]


def _flatten(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(-1)


def _repeat(vector: torch.Tensor, repeats: int) -> torch.Tensor:
    if vector.ndim != 1:
        msg = "Expected a 1D tensor."
        raise ValueError(msg)
    return _flatten(vector.repeat(repeats, 1).t())


def batch_map_coordinates(volume: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Trilinear interpolation of `volume` at `coords`."""
    batch, depth, height, width = volume.shape
    _, n_coords, _ = coords.shape

    coords = torch.stack(
        [
            coords[..., 0].clamp(0, depth - 1),
            coords[..., 1].clamp(0, height - 1),
            coords[..., 2].clamp(0, width - 1),
        ],
        dim=-1,
    )

    coords_floor = coords.floor().long()
    coords_ceil = coords.ceil().long()

    def _gather(coords_index: torch.Tensor) -> torch.Tensor:
        """
        Gather values at integer coordinates.
        coords_index : (B, N, 3) in zyx order.
        """
        batch_indices = _repeat(torch.arange(batch, device=volume.device, dtype=torch.long), n_coords)
        flat_coords = torch.stack(
            (
                batch_indices,
                _flatten(coords_index[..., 0]),
                _flatten(coords_index[..., 1]),
                _flatten(coords_index[..., 2]),
            ),
            dim=1,
        )
        flat = (
            flat_coords[:, 0] * depth * height * width
            + flat_coords[:, 1] * height * width
            + flat_coords[:, 2] * width
            + flat_coords[:, 3]
        )
        gathered = _flatten(volume).index_select(0, flat)
        return gathered.view(batch, n_coords)

    lta = coords_floor
    rbp = coords_ceil

    ltp = torch.stack((lta[..., 0], lta[..., 1], rbp[..., 2]), dim=-1)
    rtp = torch.stack((rbp[..., 0], lta[..., 1], rbp[..., 2]), dim=-1)
    rta = torch.stack((rbp[..., 0], lta[..., 1], lta[..., 2]), dim=-1)
    lba = torch.stack((lta[..., 0], rbp[..., 1], lta[..., 2]), dim=-1)
    lbp = torch.stack((lta[..., 0], rbp[..., 1], rbp[..., 2]), dim=-1)
    rba = torch.stack((rbp[..., 0], rbp[..., 1], lta[..., 2]), dim=-1)

    vals_lta = _gather(lta)
    vals_rbp = _gather(rbp)
    vals_ltp = _gather(ltp)
    vals_rtp = _gather(rtp)
    vals_rta = _gather(rta)
    vals_lba = _gather(lba)
    vals_lbp = _gather(lbp)
    vals_rba = _gather(rba)

    offset_lta = coords - lta.to(coords.dtype)
    offset_rbp = coords - rbp.to(coords.dtype)

    vals_ta = offset_lta[..., 0] * (vals_rta - vals_lta) + vals_lta
    vals_ba = offset_lta[..., 0] * (vals_rba - vals_lba) + vals_lba
    vals_tp = offset_rbp[..., 0] * (vals_rtp - vals_ltp) + vals_ltp
    vals_bp = offset_rbp[..., 0] * (vals_rbp - vals_lbp) + vals_lbp

    vals_t = offset_lta[..., 2] * (vals_tp - vals_ta) + vals_ta
    vals_b = offset_rbp[..., 2] * (vals_bp - vals_ba) + vals_ba

    mapped = offset_lta[..., 1] * (vals_b - vals_t) + vals_t
    return mapped


def generate_grid(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a `(B, D*H*W, 3)` tensor of voxel coordinates."""
    grid = np.meshgrid(range(depth), range(height), range(width), indexing="ij")
    grid = np.stack(grid, axis=-1).reshape(-1, 3).astype(np.float32)
    grid = torch.from_numpy(grid).to(device=device, dtype=dtype)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)
    return grid


def batch_map_offsets(
    volume: torch.Tensor,
    offsets: torch.Tensor,
    grid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply `offsets` (B, D, H, W, 3) to `volume` (B, D, H, W)."""
    batch, depth, height, width = volume.shape
    offsets = offsets.view(batch, -1, 3)
    if grid is None:
        grid = generate_grid(batch, depth, height, width, offsets.dtype, offsets.device)
    coords = offsets + grid
    return batch_map_coordinates(volume, coords)


@dataclass
class _GridCache:
    shape: Tuple[int, int, int, int]
    dtype: torch.dtype
    device: torch.device
    grid: torch.Tensor


class ConvOffset3D(nn.Conv3d):
    """Learn 3D offsets and resample the input using trilinear interpolation."""

    def __init__(self, in_channels: int, init_std: float = 0.01, **kwargs) -> None:
        self.filters = in_channels
        super().__init__(self.filters, self.filters * 3, kernel_size=3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_std))
        self._grid_cache: _GridCache | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        offsets = super().forward(x)
        offsets = self._reshape_to_offsets(offsets, x_shape)
        input_flat = self._reshape_to_volume(x, x_shape)
        grid = self._get_or_create_grid(x)
        sampled = batch_map_offsets(input_flat, offsets, grid=grid)
        return self._reshape_to_tensor(sampled, x_shape)

    def _get_or_create_grid(self, x: torch.Tensor) -> torch.Tensor:
        shape = (x.size(0), x.size(1), x.size(2), x.size(3))
        cache = self._grid_cache
        if cache and cache.shape == shape and cache.dtype == x.dtype and cache.device == x.device:
            return cache.grid
        grid = generate_grid(x.size(0), x.size(1), x.size(2), x.size(3), x.dtype, x.device)
        self._grid_cache = _GridCache(shape=shape, dtype=x.dtype, device=x.device, grid=grid)
        return grid

    @staticmethod
    def _init_weights(weights: torch.Tensor, std: float) -> torch.Tensor:
        fan_out = weights.size(0)
        fan_in = int(np.prod(weights.shape[1:]))
        params = np.random.normal(0.0, std, (fan_out, fan_in)).astype(np.float32)
        return torch.from_numpy(params.reshape(weights.shape))

    @staticmethod
    def _reshape_to_offsets(offsets: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        return offsets.contiguous().view(-1, x_shape[2], x_shape[3], x_shape[4], 3)

    @staticmethod
    def _reshape_to_volume(x: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        return x.contiguous().view(-1, x_shape[2], x_shape[3], x_shape[4])

    @staticmethod
    def _reshape_to_tensor(x: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        return x.contiguous().view(-1, x_shape[1], x_shape[2], x_shape[3], x_shape[4])


def deform_conv3d(in_channels: int, out_channels: int, kernel_size: int, **kwargs) -> nn.Sequential:
    """Convenience wrapper to apply learned offsets then a standard convolution."""
    return nn.Sequential(
        ConvOffset3D(out_channels),
        nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs),
    )


class LKA3d(nn.Module):
    """Large-kernel attention branch operating in 3D."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv3d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.conv_spatial = nn.Conv3d(
            channels,
            channels,
            kernel_size=7,
            stride=1,
            padding=9,
            dilation=3,
            groups=channels,
        )
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return x * attn


class LKA_Attention3d(nn.Module):
    """Apply large-kernel attention with residual projection."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj_1 = nn.Conv3d(d_model, d_model, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut
