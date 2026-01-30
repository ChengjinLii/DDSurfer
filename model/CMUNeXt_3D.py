"""3D CMUNeXt building blocks used throughout DDSurfer.

This module houses the lightweight convolutional primitives and attention
mechanisms that underpin the volumetric feature extractors used by the dual
stream networks. The implementation keeps the original numerical behaviour of
the release while providing clearer interfaces and documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from model.D_LKA_Attention import LKA_Attention3d
from model.DSCVolumeReg import T_DAModule

__all__ = [
    "ChannelSELayer",
    "AttentionGate3D",
    "conv_block_3d",
    "Residual3d",
    "CMUNeXtBlock3D",
    "CMUNeXtBlock3D_SE",
    "up_conv_3d",
    "fusion_conv_3d",
    "fusion_conv_3d_SE",
    "CMUNeXt3D_LKA_SE_TDA",
]


class ChannelSELayer(nn.Module):
    """Squeeze-and-Excitation block for 3D feature tensors."""

    def __init__(self, spatial_dims: int, in_channels: int, r: int = 4) -> None:
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            msg = "spatial_dims must be 1, 2, or 3."
            raise ValueError(msg)

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        reduced_channels = max(1, in_channels // r)
        self.excitation = nn.Sequential(
            nn.Conv3d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.excitation(self.squeeze(x))
        return x * weights


class AttentionGate3D(nn.Module):
    """3D attention gate used on U-Net skip connections."""

    def __init__(self, decoder_channels: int, encoder_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(decoder_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(encoder_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, decoder_features: torch.Tensor, encoder_features: torch.Tensor) -> torch.Tensor:
        fused = self.relu(self.W_g(decoder_features) + self.W_x(encoder_features))
        weights = self.psi(fused)
        return encoder_features * weights


class conv_block_3d(nn.Module):
    """Single 3x3 convolution + BN + ReLU used at the network stem."""

    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Residual3d(nn.Module):
    """Wrap a module to add a residual connection."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class CMUNeXtBlock3D(nn.Module):
    """Stack of depthwise separable convolutions with residual refinement."""

    def __init__(self, ch_in: int, ch_out: int, depth: int = 1, kernel_size: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(
                nn.Sequential(
                    Residual3d(
                        nn.Sequential(
                            nn.Conv3d(
                                ch_in,
                                ch_in,
                                kernel_size=(kernel_size, kernel_size, kernel_size),
                                groups=ch_in,
                                padding=kernel_size // 2,
                            ),
                            nn.GELU(),
                            nn.BatchNorm3d(ch_in),
                        )
                    ),
                    nn.Conv3d(ch_in, ch_in * 4, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm3d(ch_in * 4),
                    nn.Conv3d(ch_in * 4, ch_in, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm3d(ch_in),
                )
            )
        self.block = nn.Sequential(*layers)
        self.up = conv_block_3d(ch_in, ch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.block(x))


class CMUNeXtBlock3D_SE(nn.Module):
    """CMUNeXt block with an SE layer appended to the residual stack."""

    def __init__(self, ch_in: int, ch_out: int, depth: int = 1, kernel_size: int = 3, r: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(
                nn.Sequential(
                    Residual3d(
                        nn.Sequential(
                            nn.Conv3d(
                                ch_in,
                                ch_in,
                                kernel_size=(kernel_size, kernel_size, kernel_size),
                                groups=ch_in,
                                padding=kernel_size // 2,
                            ),
                            nn.GELU(),
                            nn.BatchNorm3d(ch_in),
                        )
                    ),
                    nn.Conv3d(ch_in, ch_in * 4, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm3d(ch_in * 4),
                    nn.Conv3d(ch_in * 4, ch_in, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm3d(ch_in),
                    ChannelSELayer(spatial_dims=3, in_channels=ch_in, r=r),
                )
            )
        self.block = nn.Sequential(*layers)
        self.up = conv_block_3d(ch_in, ch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.block(x))


class up_conv_3d(nn.Module):
    """Trilinear upsampling followed by a 3x3 convolution."""

    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class fusion_conv_3d(nn.Module):
    """Pointwise fusion block used during upsampling."""

    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm3d(ch_in),
            nn.Conv3d(ch_in, ch_out * 4, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm3d(ch_out * 4),
            nn.Conv3d(ch_out * 4, ch_out, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm3d(ch_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class fusion_conv_3d_SE(nn.Module):
    """Fusion block with an SE refinement stage."""

    def __init__(self, ch_in: int, ch_out: int, r: int = 4) -> None:
        super().__init__()
        self.original_conv_block = fusion_conv_3d(ch_in, ch_out)
        self.se_layer = ChannelSELayer(spatial_dims=3, in_channels=ch_out, r=r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.original_conv_block(x)
        return self.se_layer(x)


@dataclass(frozen=True)
class DecoderTDAConfig:
    """Configuration container for the T_DAModule blocks."""

    reduction_ratio: int = 8
    pool_types: Sequence[str] = ("avg", "max")
    no_spatial_depth: bool = False


class CMUNeXt3D_LKA_SE_TDA(nn.Module):
    """CMUNeXt encoder-decoder augmented with LKA and TDA modules."""

    def __init__(
        self,
        input_channel: int = 3,
        num_classes: int = 1,
        dims: Sequence[int] = (16, 32, 64, 128, 256),
        depths: Sequence[int] = (1, 1, 1, 2, 1),
        kernels: Sequence[int] = (3, 3, 5, 5, 5),
        se_reduction: int = 4,
        tda_cfg: DecoderTDAConfig | None = None,
    ) -> None:
        super().__init__()
        if len(dims) != 5 or len(depths) != 5 or len(kernels) != 5:
            msg = "dims, depths, and kernels must all have length 5."
            raise ValueError(msg)

        tda_cfg = tda_cfg or DecoderTDAConfig()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.stem = conv_block_3d(ch_in=input_channel, ch_out=dims[0])

        self.encoder1 = CMUNeXtBlock3D_SE(dims[0], dims[0], depth=depths[0], kernel_size=kernels[0], r=se_reduction)
        self.encoder2 = CMUNeXtBlock3D_SE(dims[0], dims[1], depth=depths[1], kernel_size=kernels[1], r=se_reduction)
        self.encoder3 = CMUNeXtBlock3D_SE(dims[1], dims[2], depth=depths[2], kernel_size=kernels[2], r=se_reduction)
        self.encoder4 = CMUNeXtBlock3D_SE(dims[2], dims[3], depth=depths[3], kernel_size=kernels[3], r=se_reduction)
        self.encoder5 = CMUNeXtBlock3D_SE(dims[3], dims[4], depth=depths[4], kernel_size=kernels[4], r=se_reduction)

        self.lka_attention = LKA_Attention3d(d_model=dims[4])

        self.up5 = up_conv_3d(dims[4], dims[3])
        self.up_conv5 = fusion_conv_3d_SE(dims[3] * 2, dims[3], r=se_reduction)
        self.tda5 = T_DAModule(
            gate_channels=dims[3],
            reduction_ratio=tda_cfg.reduction_ratio,
            pool_type=list(tda_cfg.pool_types),
            no_spatial_depth=tda_cfg.no_spatial_depth,
        )

        self.up4 = up_conv_3d(dims[3], dims[2])
        self.up_conv4 = fusion_conv_3d_SE(dims[2] * 2, dims[2], r=se_reduction)
        self.tda4 = T_DAModule(
            gate_channels=dims[2],
            reduction_ratio=tda_cfg.reduction_ratio,
            pool_type=list(tda_cfg.pool_types),
            no_spatial_depth=tda_cfg.no_spatial_depth,
        )

        self.up3 = up_conv_3d(dims[2], dims[1])
        self.up_conv3 = fusion_conv_3d_SE(dims[1] * 2, dims[1], r=se_reduction)
        self.tda3 = T_DAModule(
            gate_channels=dims[1],
            reduction_ratio=tda_cfg.reduction_ratio,
            pool_type=list(tda_cfg.pool_types),
            no_spatial_depth=tda_cfg.no_spatial_depth,
        )

        self.up2 = up_conv_3d(dims[1], dims[0])
        self.up_conv2 = fusion_conv_3d_SE(dims[0] * 2, dims[0], r=se_reduction)
        self.tda2 = T_DAModule(
            gate_channels=dims[0],
            reduction_ratio=tda_cfg.reduction_ratio,
            pool_type=list(tda_cfg.pool_types),
            no_spatial_depth=tda_cfg.no_spatial_depth,
        )

        self.output_head = nn.Conv3d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.Maxpool(x1))
        x3 = self.encoder3(self.Maxpool(x2))
        x4 = self.encoder4(self.Maxpool(x3))
        x5 = self.encoder5(self.Maxpool(x4))

        x5_attended = self.lka_attention(x5)

        d5 = self.tda5(self.up_conv5(torch.cat((x4, self.up5(x5_attended)), dim=1)))
        d4 = self.tda4(self.up_conv4(torch.cat((x3, self.up4(d5)), dim=1)))
        d3 = self.tda3(self.up_conv3(torch.cat((x2, self.up3(d4)), dim=1)))
        d2 = self.tda2(self.up_conv2(torch.cat((x1, self.up2(d3)), dim=1)))

        return self.output_head(d2)
