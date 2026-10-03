"""DDSurfer dual-stream temporal attention network."""

from __future__ import annotations

from functools import partial
from typing import Sequence

import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F

from model.cmunext import (AttentionGate3D, CMUNeXtBlock3D, CMUNeXtSEBlock3D,
                           ConvBlock3D, FusionBlock3D, UpsampleBlock3D)
from model.attention import LKAAttention3D
from model.refinement import (GeometryAwareLKA3D,
                              ResidualGate, ResidualSkipGate3D,
                              cross_gate_options, replace_batchnorm, resolve_refinement)

# ------------------------- Building Blocks -------------------------
class CrossStreamFusion(nn.Module):
    """Voxelwise cross-stream gating, optionally with bounded residual modulation."""

    def __init__(self, channels_a: int, channels_b: int, reduction: int = 4,
                 residual: bool = False, min_hidden: int = 1,
                 max_hidden: int | None = None, gain_init: float = 0.1,
                 max_gain: float = 0.25) -> None:
        super().__init__()
        if reduction < 1 or min_hidden < 1 or (max_hidden is not None and max_hidden < min_hidden):
            raise ValueError("Invalid cross-stream gate hidden width configuration")
        self.residual = residual
        inter_channels_a = max(min_hidden, channels_a // reduction)
        inter_channels_b = max(min_hidden, channels_b // reduction)
        if max_hidden is not None:
            inter_channels_a = min(max_hidden, inter_channels_a)
            inter_channels_b = min(max_hidden, inter_channels_b)
        self.gate_generator_A = nn.Sequential(
            nn.Conv3d(channels_b, inter_channels_a, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels_a, channels_a, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gate_generator_B = nn.Sequential(
            nn.Conv3d(channels_a, inter_channels_b, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels_b, channels_b, kernel_size=1),
            nn.Sigmoid(),
        )
        if residual:
            self.modulation_a = ResidualGate(gain_init, max_gain)
            self.modulation_b = ResidualGate(gain_init, max_gain)
            for generator in (self.gate_generator_A, self.gate_generator_B):
                nn.init.zeros_(generator[2].weight)
                nn.init.zeros_(generator[2].bias)

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor,
                return_streams: bool = False):
        weights_a = self.gate_generator_A(features_b)
        weights_b = self.gate_generator_B(features_a)
        if self.residual:
            refined_a = self.modulation_a(features_a, weights_a)
            refined_b = self.modulation_b(features_b, weights_b)
        else:
            refined_a = features_a * weights_a
            refined_b = features_b * weights_b
        if return_streams:
            # Encoder paths remain separate; avoid a full-resolution concat/split copy.
            return refined_a, refined_b
        return torch.cat((refined_a, refined_b), dim=1)


# ------------------------- Velocity Field Backbone -------------------------
class VelocityFieldNet(nn.Module):
    """Dual-stream CMUNeXt encoder-decoder that predicts cascaded velocity fields."""

    def __init__(
        self,
        input_channel: int = 5,
        dims: list[int] | tuple[int, ...] = (16, 32, 64, 128, 256),
        depths: list[int] | tuple[int, ...] = (1, 1, 1, 6, 3),
        kernels: list[int] | tuple[int, ...] = (3, 3, 7, 7, 7),
        M: int = 2,
        R: int = 3,
        kernel_size: int = 3,
        inshape: Sequence[int] = (112, 224, 176),
        refinement: dict | None = None,
    ) -> None:
        super().__init__()
        self.M, self.R = int(M), int(R)
        if len(dims) != 5 or len(depths) != 5 or len(kernels) != 5:
            raise ValueError("dims, depths and kernels must have five stages")
        if any(d < 3 or int(d) != d for d in dims) or any(n < 1 or int(n) != n for n in depths):
            raise ValueError("Widths must be integers >= 3 and depths positive integers")
        if any(k < 1 or k % 2 == 0 or int(k) != k for k in kernels):
            raise ValueError("Stage kernels must be positive odd integers")
        if input_channel < 2:
            raise ValueError("Dual streams require FA and at least one other channel")
        self.input_channel = input_channel
        self.activation_checkpointing = False
        self.inshape = tuple(inshape)
        self.encoder_fusion_enabled = refinement is not None and refinement["encoder_fusion"]
        if len(inshape) != 3 or any(s < 16 or s % 16 or int(s) != s for s in inshape):
            raise ValueError("Each spatial dimension must be a positive multiple of 16")
        encoder_block = CMUNeXtSEBlock3D if refinement is None or refinement["use_se"] else CMUNeXtBlock3D
        gate_policy = refinement["cross_gate_policy"] if refinement is not None else "standard"

        def fusion(channels_a, channels_b, stage):
            return CrossStreamFusion(
                channels_a, channels_b, residual=refinement is not None and refinement["residual_gates"],
                **cross_gate_options(gate_policy, stage))

        skip_gate = (partial(ResidualSkipGate3D, norm_type=refinement["norm_type"])
                     if refinement is not None and refinement["residual_gates"] else AttentionGate3D)

        self.dims_A = [int(d * 0.4) for d in dims]
        self.dims_B = [d - a for d, a in zip(dims, self.dims_A)]

        # Encoder streams
        self.stem_A = ConvBlock3D(1, self.dims_A[0])
        self.encoder1_A = encoder_block(self.dims_A[0], self.dims_A[0], depth=depths[0], kernel_size=kernels[0])
        self.encoder2_A = encoder_block(self.dims_A[0], self.dims_A[1], depth=depths[1], kernel_size=kernels[1])
        self.encoder3_A = encoder_block(self.dims_A[1], self.dims_A[2], depth=depths[2], kernel_size=kernels[2])
        self.encoder4_A = encoder_block(self.dims_A[2], self.dims_A[3], depth=depths[3], kernel_size=kernels[3])
        self.encoder5_A = encoder_block(self.dims_A[3], self.dims_A[4], depth=depths[4], kernel_size=kernels[4])

        self.stem_B = ConvBlock3D(input_channel - 1, self.dims_B[0])
        self.encoder1_B = encoder_block(self.dims_B[0], self.dims_B[0], depth=depths[0], kernel_size=kernels[0])
        self.encoder2_B = encoder_block(self.dims_B[0], self.dims_B[1], depth=depths[1], kernel_size=kernels[1])
        self.encoder3_B = encoder_block(self.dims_B[1], self.dims_B[2], depth=depths[2], kernel_size=kernels[2])
        self.encoder4_B = encoder_block(self.dims_B[2], self.dims_B[3], depth=depths[3], kernel_size=kernels[3])
        self.encoder5_B = encoder_block(self.dims_B[3], self.dims_B[4], depth=depths[4], kernel_size=kernels[4])

        # Cross-stream fusion and bottleneck
        if refinement is None or self.encoder_fusion_enabled:
            # Cross-stream fusion before downsampling.
            for index in range(4):
                setattr(self, f"fusion{index + 1}", fusion(self.dims_A[index], self.dims_B[index], index + 1))
        self.fusion5 = fusion(self.dims_A[4], self.dims_B[4], 5)

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.lka_attention = (GeometryAwareLKA3D(dims[4], [s // 16 for s in inshape], refinement["norm_type"])
                              if refinement is not None and refinement["geometry_aware_lka"]
                              else LKAAttention3D(d_model=dims[4]))

        # Decoder with attention gates
        self.Up5_A = UpsampleBlock3D(self.dims_A[4], self.dims_A[3])
        self.Up5_B = UpsampleBlock3D(self.dims_B[4], self.dims_B[3])
        self.Att5_A = skip_gate(self.dims_A[3], self.dims_A[3], self.dims_A[2])
        self.Att5_B = skip_gate(self.dims_B[3], self.dims_B[3], self.dims_B[2])
        self.Up_conv5_A = FusionBlock3D(self.dims_A[3] * 2, self.dims_A[3])
        self.Up_conv5_B = FusionBlock3D(self.dims_B[3] * 2, self.dims_B[3])
        self.decoder_fusion4 = fusion(self.dims_A[3], self.dims_B[3], 4)

        self.Up4_A = UpsampleBlock3D(self.dims_A[3], self.dims_A[2])
        self.Up4_B = UpsampleBlock3D(self.dims_B[3], self.dims_B[2])
        self.Att4_A = skip_gate(self.dims_A[2], self.dims_A[2], self.dims_A[1])
        self.Att4_B = skip_gate(self.dims_B[2], self.dims_B[2], self.dims_B[1])
        self.Up_conv4_A = FusionBlock3D(self.dims_A[2] * 2, self.dims_A[2])
        self.Up_conv4_B = FusionBlock3D(self.dims_B[2] * 2, self.dims_B[2])
        self.decoder_fusion3 = fusion(self.dims_A[2], self.dims_B[2], 3)

        self.Up3_A = UpsampleBlock3D(self.dims_A[2], self.dims_A[1])
        self.Up3_B = UpsampleBlock3D(self.dims_B[2], self.dims_B[1])
        self.Att3_A = skip_gate(self.dims_A[1], self.dims_A[1], self.dims_A[0])
        self.Att3_B = skip_gate(self.dims_B[1], self.dims_B[1], self.dims_B[0])
        self.Up_conv3_A = FusionBlock3D(self.dims_A[1] * 2, self.dims_A[1])
        self.Up_conv3_B = FusionBlock3D(self.dims_B[1] * 2, self.dims_B[1])
        self.decoder_fusion2 = fusion(self.dims_A[1], self.dims_B[1], 2)

        self.Up2_A = UpsampleBlock3D(self.dims_A[1], self.dims_A[0])
        self.Up2_B = UpsampleBlock3D(self.dims_B[1], self.dims_B[0])
        self.Att2_A = skip_gate(self.dims_A[0], self.dims_A[0], max(1, self.dims_A[0] // 2))
        self.Att2_B = skip_gate(self.dims_B[0], self.dims_B[0], max(1, self.dims_B[0] // 2))
        self.Up_conv2_A = FusionBlock3D(self.dims_A[0] * 2, self.dims_A[0])
        self.Up_conv2_B = FusionBlock3D(self.dims_B[0] * 2, self.dims_B[0])

        # Cascaded velocity heads
        self.flow1 = nn.Conv3d(dims[2], 3 * M, kernel_size=kernel_size, padding=kernel_size // 2)
        self.flow2 = nn.Conv3d(dims[1] + 3 * M, 3 * M, kernel_size=kernel_size, padding=kernel_size // 2)
        self.flow3 = nn.Conv3d(dims[0] + 3 * M, 3 * M, kernel_size=kernel_size, padding=kernel_size // 2)
        for conv in (self.flow1, self.flow2, self.flow3):
            nn.init.normal_(conv.weight, 0.0, 1e-5)
            nn.init.constant_(conv.bias, 0.0)

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        if refinement is not None:
            replace_batchnorm(self, refinement["norm_type"])

    def _fuse_encoder(self, stage, features_a, features_b):
        if not self.encoder_fusion_enabled:
            return features_a, features_b
        module = getattr(self, f"fusion{stage}")
        return self._run(lambda a, b: module(a, b, return_streams=True), features_a, features_b)

    def _run(self, function, *inputs):
        # Recompute intermediate activations when checkpointing is enabled.
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(function, *inputs, use_reentrant=False, preserve_rng_state=True)
        return function(*inputs)

    def forward(self, x: torch.Tensor, return_context: bool = False):
        if x.ndim != 5 or x.shape[0] != 1 or x.shape[1] != self.input_channel:
            raise ValueError("VF backbone requires [1,C_in,X,Y,Z]")
        if tuple(x.shape[2:]) != self.inshape:
            raise ValueError("VF input differs from its planned spatial shape")
        stream_a = x[:, 0:1]
        stream_b = x[:, 1:]

        x1_a = self._run(self.encoder1_A, self._run(self.stem_A, stream_a))
        x1_b = self._run(self.encoder1_B, self._run(self.stem_B, stream_b))
        x1_a, x1_b = self._fuse_encoder(1, x1_a, x1_b)
        x2_a = self._run(self.encoder2_A, self.Maxpool(x1_a))
        x2_b = self._run(self.encoder2_B, self.Maxpool(x1_b))
        x2_a, x2_b = self._fuse_encoder(2, x2_a, x2_b)
        x3_a = self._run(self.encoder3_A, self.Maxpool(x2_a))
        x3_b = self._run(self.encoder3_B, self.Maxpool(x2_b))
        x3_a, x3_b = self._fuse_encoder(3, x3_a, x3_b)
        x4_a = self._run(self.encoder4_A, self.Maxpool(x3_a))
        x4_b = self._run(self.encoder4_B, self.Maxpool(x3_b))
        x4_a, x4_b = self._fuse_encoder(4, x4_a, x4_b)
        x5_a = self._run(self.encoder5_A, self.Maxpool(x4_a))
        x5_b = self._run(self.encoder5_B, self.Maxpool(x4_b))

        bottleneck = self._run(self.lka_attention, self._run(self.fusion5, x5_a, x5_b))
        d5_a_in, d5_b_in = torch.split(bottleneck, [self.dims_A[4], self.dims_B[4]], dim=1)

        d5_a = self._run(self.Up5_A, d5_a_in)
        d5_b = self._run(self.Up5_B, d5_b_in)
        x4_a_att = self._run(self.Att5_A, d5_a, x4_a)
        x4_b_att = self._run(self.Att5_B, d5_b, x4_b)
        d4_a = self._run(self.Up_conv5_A, torch.cat([x4_a_att, d5_a], dim=1))
        d4_b = self._run(self.Up_conv5_B, torch.cat([x4_b_att, d5_b], dim=1))
        d4_fused = self._run(self.decoder_fusion4, d4_a, d4_b)
        d4_a_out, d4_b_out = torch.split(d4_fused, [self.dims_A[3], self.dims_B[3]], dim=1)

        d4_a_up = self._run(self.Up4_A, d4_a_out)
        d4_b_up = self._run(self.Up4_B, d4_b_out)
        x3_a_att = self._run(self.Att4_A, d4_a_up, x3_a)
        x3_b_att = self._run(self.Att4_B, d4_b_up, x3_b)
        d3_a = self._run(self.Up_conv4_A, torch.cat([x3_a_att, d4_a_up], dim=1))
        d3_b = self._run(self.Up_conv4_B, torch.cat([x3_b_att, d4_b_up], dim=1))
        d3_fused = self._run(self.decoder_fusion3, d3_a, d3_b)
        vf1_small = self._run(self.flow1, d3_fused)

        d3_a_out, d3_b_out = torch.split(d3_fused, [self.dims_A[2], self.dims_B[2]], dim=1)
        d3_a_up = self._run(self.Up3_A, d3_a_out)
        d3_b_up = self._run(self.Up3_B, d3_b_out)
        x2_a_att = self._run(self.Att3_A, d3_a_up, x2_a)
        x2_b_att = self._run(self.Att3_B, d3_b_up, x2_b)
        d2_a = self._run(self.Up_conv3_A, torch.cat([x2_a_att, d3_a_up], dim=1))
        d2_b = self._run(self.Up_conv3_B, torch.cat([x2_b_att, d3_b_up], dim=1))
        d2_fused = self._run(self.decoder_fusion2, d2_a, d2_b)

        vf1_medium = self.up(vf1_small)
        vf2_medium = vf1_medium + self._run(self.flow2, torch.cat([d2_fused, vf1_medium], dim=1))

        d2_a_out, d2_b_out = torch.split(d2_fused, [self.dims_A[1], self.dims_B[1]], dim=1)
        d2_a_up = self._run(self.Up2_A, d2_a_out)
        d2_b_up = self._run(self.Up2_B, d2_b_out)
        x1_a_att = self._run(self.Att2_A, d2_a_up, x1_a)
        x1_b_att = self._run(self.Att2_B, d2_b_up, x1_b)
        d1_a = self._run(self.Up_conv2_A, torch.cat([x1_a_att, d2_a_up], dim=1))
        d1_b = self._run(self.Up_conv2_B, torch.cat([x1_b_att, d2_b_up], dim=1))
        d1_fused = torch.cat([d1_a, d1_b], dim=1)

        vf2_full = self.up(vf2_medium)
        vf3_full = vf2_full + self._run(self.flow3, torch.cat([d1_fused, vf2_full], dim=1))
        vf1_full = self.up(vf1_medium)

        vf1 = vf1_full.reshape(self.M, 3, *vf1_full.shape[2:])
        vf2 = vf2_full.reshape(self.M, 3, *vf2_full.shape[2:])
        vf3 = vf3_full.reshape(self.M, 3, *vf3_full.shape[2:])

        if self.R == 3:
            fields = torch.cat([vf1, vf2, vf3], dim=0)
        elif self.R == 2:
            fields = torch.cat([vf2, vf3], dim=0)
        else:
            fields = vf3
        if return_context:
            return fields, F.adaptive_avg_pool3d(bottleneck, 1).flatten(1)
        return fields

# ------------------------- Temporal Attention -------------------------
class TemporalAttentionNet(nn.Module):
    """Temporal attention predictor for weighting stationary velocity fields."""

    def __init__(self, hidden_channels: int = 16, M: int = 2, R: int = 3,
                 context_channels: int | None = None) -> None:
        super().__init__()
        self.context_channels = context_channels
        self.fc1 = nn.Linear(1, hidden_channels * 4)
        self.fc2 = nn.Linear(hidden_channels * 4, hidden_channels * 8)
        self.fc3 = nn.Linear(hidden_channels * 8, hidden_channels * 8)
        self.fc4 = nn.Linear(hidden_channels * 8, hidden_channels * 4)
        self.fc5 = nn.Linear(hidden_channels * 4, M * R)
        if context_channels is not None:
            self.context_film = nn.Sequential(nn.LayerNorm(context_channels),
                                              nn.Linear(context_channels, hidden_channels * 8))
            # Begin with the shared time policy; learn subject-specific corrections.
            nn.init.zeros_(self.context_film[1].weight)
            nn.init.zeros_(self.context_film[1].bias)

    def forward(self, t: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        out = F.leaky_relu(self.fc1(t), 0.2)
        if self.context_channels is not None:
            if context is None or context.shape != (1, self.context_channels):
                raise ValueError("Conditional attention requires one pooled image-feature vector")
            gamma, beta = self.context_film(context).chunk(2, dim=-1)
            out = out * (1.0 + 0.1 * gamma.tanh()) + 0.1 * beta.tanh()
        out = F.leaky_relu(self.fc2(out), 0.2)
        out = F.leaky_relu(self.fc3(out), 0.2)
        out = F.leaky_relu(self.fc4(out), 0.2)
        return F.softmax(self.fc5(out), dim=-1)

# ------------------------- TANet -------------------------

class TANet(nn.Module):
    """Temporal attention network with RK4 integration over stationary fields."""

    def __init__(
        self,
        C_in: int = 5,
        C_hid: Sequence[int] = (16, 32, 64, 128, 256),
        inshape: Sequence[int] = (112, 224, 176),
        depths: Sequence[int] = (1, 1, 1, 6, 3),
        kernels: Sequence[int] = (3, 3, 7, 7, 7),
        step_size: float = 0.02,
        M: int = 2,
        R: int = 3,
        device: str = "cuda:0",
        refinement: dict | None = None,
    ) -> None:
        super().__init__()
        self.M = int(M)
        self.R = int(R)
        if not 0 < step_size <= 1 or abs(round(1.0 / step_size) * step_size - 1.0) > 1e-6:
            raise ValueError("step_size must divide the integration interval [0,1]")
        if R not in (1, 2, 3) or M < 1 or int(M) != M:
            raise ValueError("Expected M >= 1 and R in (1,2,3)")
        self.inshape = tuple(inshape)
        self.refinement = resolve_refinement(refinement)
        self.conditioned_attention = self.refinement is not None and self.refinement["conditioned_attention"]
        self.rk4_stage_time = self.refinement is not None and self.refinement["rk4_stage_time"]
        self.vf_net = VelocityFieldNet(C_in, C_hid, depths, kernels, M=self.M, R=self.R,
                                       inshape=inshape, refinement=self.refinement)
        self.att_net = TemporalAttentionNet(hidden_channels=16, M=self.M, R=self.R,
                                           context_channels=C_hid[-1] if self.conditioned_attention else None)

        self.h = float(step_size)
        self.num_steps = max(1, int(round(1.0 / self.h)))
        # Nonpersistent buffers preserve strict loading of release state_dicts.
        times = (torch.arange(2 * self.num_steps + 1)[:, None] * (self.h / 2)
                 if self.rk4_stage_time else torch.arange(self.num_steps)[:, None] * self.h)
        self.register_buffer("timesteps", times, persistent=False)
        self.register_buffer("scale", torch.as_tensor(inshape, dtype=torch.float32)[None, None, :] - 1.0, persistent=False)

    def forward(self, vertices: torch.Tensor, volumes: torch.Tensor, return_extras: bool = False):
        if vertices.ndim != 3 or vertices.shape[-1] != 3 or volumes.ndim != 5:
            raise ValueError("Expected vertices [1,N,3] and volumes [1,C,X,Y,Z]")
        if vertices.shape[0] != 1 or volumes.shape[0] != 1:
            raise ValueError("DDSurfer TANet requires batch_size=1")
        if tuple(volumes.shape[2:]) != self.inshape:
            raise ValueError("Volume shape does not match the model's crop-voxel geometry")
        context = None
        if self.conditioned_attention:
            svfs_all, context = self.vf_net(volumes, return_context=True)
        else:
            svfs_all = self.vf_net(volumes)
        expected = self.M * self.R
        if svfs_all.dim() != 5:
            msg = f"vf_net output expects 5D, got {svfs_all.shape}"
            raise RuntimeError(msg)
        if svfs_all.shape[0] == 3 and svfs_all.shape[-1] == expected:
            svfs_all = svfs_all.permute(4, 0, 1, 2, 3).contiguous()
        elif svfs_all.shape[0] != expected or svfs_all.shape[1] != 3:
            msg = f"Unexpected SVF shape {svfs_all.shape}, expect [MR,3,X,Y,Z] with MR={expected}"
            raise RuntimeError(msg)

        # Keep velocity interpolation and all 50 RK4 steps in FP32 under AMP.
        with torch.autocast(device_type=volumes.device.type, enabled=False):
            return self._integrate(vertices.float(), svfs_all.float(), return_extras,
                                   context.float() if context is not None else None)

    def _integrate(self, vertices: torch.Tensor, svfs_all: torch.Tensor, return_extras: bool,
                   context: torch.Tensor | None = None):
        attention = self.att_net(self.timesteps.float(), context)[..., None, None]

        current_vertices = vertices
        trajectory_stats = [] if return_extras else None
        for step in range(self.num_steps):
            weights = attention[2 * step if self.rk4_stage_time else step]
            middle_weights = attention[2 * step + 1] if self.rk4_stage_time else weights
            end_weights = attention[2 * step + 2] if self.rk4_stage_time else weights

            def sample_field(sample_vertices: torch.Tensor, stage_weights: torch.Tensor) -> torch.Tensor:
                raw = self.interpolate(sample_vertices, svfs_all)
                return (stage_weights.view(-1, 1, 1) * raw).sum(0, keepdim=True)

            k1 = sample_field(current_vertices, weights)
            k2 = sample_field(current_vertices + self.h * 0.5 * k1, middle_weights)
            k3 = sample_field(current_vertices + self.h * 0.5 * k2, middle_weights)
            k4 = sample_field(current_vertices + self.h * k3, end_weights)
            velocity = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            current_vertices = current_vertices + self.h * velocity

            if return_extras:
                speed = torch.linalg.vector_norm(velocity[0], dim=-1)
                trajectory_stats.append(
                    {
                        "t": float((step + 1) * self.h),
                        "speed_mean": float(speed.mean()),
                        "speed_p95": float(torch.quantile(speed, 0.95)),
                        "speed_max": float(speed.max()),
                    }
                )

        if not return_extras:
            return current_vertices

        per_level = {}
        offset = 0
        for level in range(self.R):
            per_level[f"vf{4 - self.R + level}"] = svfs_all[offset : offset + self.M].detach()
            offset += self.M

        extras = {
            "svfs_all": svfs_all.detach(),
            "per_level": per_level,
            "att_weights": attention.squeeze(-1).squeeze(-1).detach(),
            "att_times": self.timesteps.detach(),
            "gate_stats": {name: module.last_gate_stats for name, module in self.named_modules()
                           if isinstance(module, ResidualGate) and module.collect_diagnostics},
            "traj_stats": trajectory_stats,
        }
        return current_vertices, extras

    def enable_activation_checkpointing(self, enabled: bool = True):
        if enabled and any(isinstance(module, nn.BatchNorm3d) for module in self.vf_net.modules()):
            raise ValueError("Checkpoint recomputation must not double-update BatchNorm running statistics")
        self.vf_net.activation_checkpointing = enabled

    def set_gate_diagnostics(self, enabled: bool = True):
        """Enable scalar gate statistics only during debugging, not each training step."""
        for module in self.modules():
            if isinstance(module, ResidualGate):
                module.collect_diagnostics = enabled
                module.last_gate_stats = None

    def interpolate(self, vertices: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        coords = 2.0 * vertices / self.scale - 1.0
        coords = coords.repeat(fields.shape[0], 1, 1)
        coords = coords[:, :, None, None].flip(-1)
        sampled = F.grid_sample(fields, coords, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled[..., 0, 0].permute(0, 2, 1)
