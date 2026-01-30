"""Depthwise separable cost-volume regularisation blocks."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DSConv3d",
    "DSDeConv3d",
    "ChannelGate",
    "SpatialDepthGate",
    "T_DAModule",
    "DSCVolumeReg",
]


class DSConv3d(nn.Module):
    """Depthwise separable 3D convolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm3d(in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.depthwise_bn(self.depthwise(x)), inplace=True)
        x = F.relu(self.pointwise_bn(self.pointwise(x)), inplace=True)
        return x


class DSDeConv3d(nn.Module):
    """Depthwise separable 3D transposed convolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.depthwise = nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm3d(in_channels)
        self.pointwise = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.depthwise_bn(self.depthwise(x)), inplace=True)
        x = F.relu(self.pointwise_bn(self.pointwise(x)), inplace=True)
        return x


class Flatten(nn.Module):
    """Flatten spatial dimensions to a single vector per sample."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    """Channel attention via global pooling and an MLP."""

    def __init__(self, channels: int, reduction_ratio: int = 16, pool_types: Sequence[str] = ("avg", "max")) -> None:
        super().__init__()
        self.pool_types = tuple(pool_types)
        reduced = max(1, channels // reduction_ratio)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled: torch.Tensor | None = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                pooled_tensor = F.avg_pool3d(x, kernel_size=x.shape[2:])
            elif pool_type == "max":
                pooled_tensor = F.max_pool3d(x, kernel_size=x.shape[2:])
            else:
                raise ValueError(f"Unsupported pool type: {pool_type}")

            channel_attention = self.mlp(pooled_tensor)
            pooled = channel_attention if pooled is None else pooled + channel_attention

        scale = torch.sigmoid(pooled).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * scale.expand_as(x)


class BasicConv(nn.Module):
    """Utility convolution with optional BN and ReLU."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: Sequence[int],
        stride: int = 1,
        padding: Sequence[int] | int = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        relu: bool = True,
        bn: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    """Compute max and average activation maps and concatenate them."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pool = torch.max(x, dim=1, keepdim=True).values
        mean_pool = torch.mean(x, dim=1, keepdim=True)
        return torch.cat((max_pool, mean_pool), dim=1)


class SpatialDepthGate(nn.Module):
    """Spatial + depth attention using factorised convolutions."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.channel_pool = ChannelPool()
        self.channel_conv = BasicConv(
            2,
            1,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, padding, padding),
            relu=False,
        )
        self.depth_conv = BasicConv(
            1,
            1,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            relu=False,
        )
        self.overall_conv = BasicConv(
            1,
            1,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            padding=padding,
            relu=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.channel_pool(x)
        pooled = self.channel_conv(pooled)
        pooled = self.depth_conv(pooled)
        pooled = self.overall_conv(pooled)
        scale = torch.sigmoid(pooled)
        return x * scale


class T_DAModule(nn.Module):
    """3D attention module combining channel and spatial-depth gates."""

    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int = 16,
        pool_type: Sequence[str] = ("avg", "max"),
        no_spatial_depth: bool = False,
    ) -> None:
        super().__init__()
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio, pool_type)
        self.no_spatial_depth = no_spatial_depth
        self.spatial_depth_gate = None if no_spatial_depth else SpatialDepthGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_gate(x)
        if not self.no_spatial_depth and self.spatial_depth_gate is not None:
            x = self.spatial_depth_gate(x)
        return x


class DSCVolumeReg(nn.Module):
    """Depthwise separable cost-volume regulariser with attention refinement."""

    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.base_channels = base_channels

        self.conv0_1 = DSConv3d(in_channels, base_channels, stride=1)

        self.conv1_0 = DSConv3d(in_channels, base_channels * 2, stride=2)
        self.conv1_1 = DSConv3d(base_channels * 2, base_channels * 2, stride=1)

        self.conv2_0 = DSConv3d(base_channels * 2, base_channels * 4, stride=2)
        self.conv2_1 = DSConv3d(base_channels * 4, base_channels * 4, stride=1)

        self.conv3_0 = DSConv3d(base_channels * 4, base_channels * 8, stride=2)
        self.conv3_1 = DSConv3d(base_channels * 8, base_channels * 8, stride=1)

        self.conv4_0 = DSDeConv3d(base_channels * 8, base_channels * 4, stride=2)
        self.conv5_0 = DSDeConv3d(base_channels * 4, base_channels * 2, stride=2)
        self.conv6_0 = DSDeConv3d(base_channels * 2, base_channels, stride=2)

        self.attention_block_1 = T_DAModule(base_channels * 4, reduction_ratio=8, pool_type=("avg", "max"))
        self.attention_block_2 = T_DAModule(base_channels * 2, reduction_ratio=8, pool_type=("avg", "max"))
        self.attention_block_3 = T_DAModule(base_channels, reduction_ratio=8, pool_type=("avg", "max"))

        self.output_conv = nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv0_1 = self.conv0_1(x)

        conv1_0 = self.conv1_0(x)
        conv2_0 = self.conv2_0(conv1_0)
        conv3_0 = self.conv3_0(conv2_0)

        conv1_1 = self.conv1_1(conv1_0)
        conv2_1 = self.conv2_1(conv2_0)
        conv3_1 = self.conv3_1(conv3_0)

        conv4_0 = self.conv4_0(conv3_1)
        conv5_0 = self.conv5_0(self.attention_block_1(conv4_0 + conv2_1))
        conv6_0 = self.conv6_0(self.attention_block_2(conv5_0 + conv1_1))

        output = self.output_conv(self.attention_block_3(conv6_0 + conv0_1))
        return output
