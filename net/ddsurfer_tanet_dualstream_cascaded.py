"""TANet dual-stream architecture using CMUNeXt backbones and large-kernel attention."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.CMUNeXt_3D import AttentionGate3D, CMUNeXtBlock3D_SE, conv_block_3d, fusion_conv_3d, up_conv_3d
from model.D_LKA_Attention import LKA_Attention3d

# ------------------------- Building Blocks -------------------------
class CrossAttentionFusionModule(nn.Module):
    """Bidirectional channel attention to fuse dual streams."""

    def __init__(self, channels_a: int, channels_b: int, reduction: int = 4) -> None:
        super().__init__()
        inter_channels_a = max(1, channels_a // reduction)
        inter_channels_b = max(1, channels_b // reduction)
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

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        refined_a = features_a * self.gate_generator_A(features_b)
        refined_b = features_b * self.gate_generator_B(features_a)
        return torch.cat((refined_a, refined_b), dim=1)


# ------------------------- Velocity Field Backbone -------------------------
class CMUNeXt_VFNet_Final(nn.Module):
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
    ) -> None:
        super().__init__()
        self.M, self.R = int(M), int(R)

        self.dims_A = [int(d * 0.4) for d in dims]
        self.dims_B = [d - a for d, a in zip(dims, self.dims_A)]

        # Encoder streams
        self.stem_A = conv_block_3d(1, self.dims_A[0])
        self.encoder1_A = CMUNeXtBlock3D_SE(self.dims_A[0], self.dims_A[0], depth=depths[0], kernel_size=kernels[0])
        self.encoder2_A = CMUNeXtBlock3D_SE(self.dims_A[0], self.dims_A[1], depth=depths[1], kernel_size=kernels[1])
        self.encoder3_A = CMUNeXtBlock3D_SE(self.dims_A[1], self.dims_A[2], depth=depths[2], kernel_size=kernels[2])
        self.encoder4_A = CMUNeXtBlock3D_SE(self.dims_A[2], self.dims_A[3], depth=depths[3], kernel_size=kernels[3])
        self.encoder5_A = CMUNeXtBlock3D_SE(self.dims_A[3], self.dims_A[4], depth=depths[4], kernel_size=kernels[4])

        self.stem_B = conv_block_3d(input_channel - 1, self.dims_B[0])
        self.encoder1_B = CMUNeXtBlock3D_SE(self.dims_B[0], self.dims_B[0], depth=depths[0], kernel_size=kernels[0])
        self.encoder2_B = CMUNeXtBlock3D_SE(self.dims_B[0], self.dims_B[1], depth=depths[1], kernel_size=kernels[1])
        self.encoder3_B = CMUNeXtBlock3D_SE(self.dims_B[1], self.dims_B[2], depth=depths[2], kernel_size=kernels[2])
        self.encoder4_B = CMUNeXtBlock3D_SE(self.dims_B[2], self.dims_B[3], depth=depths[3], kernel_size=kernels[3])
        self.encoder5_B = CMUNeXtBlock3D_SE(self.dims_B[3], self.dims_B[4], depth=depths[4], kernel_size=kernels[4])

        # Cross-stream fusion and bottleneck
        self.fusion1 = CrossAttentionFusionModule(self.dims_A[0], self.dims_B[0])
        self.fusion2 = CrossAttentionFusionModule(self.dims_A[1], self.dims_B[1])
        self.fusion3 = CrossAttentionFusionModule(self.dims_A[2], self.dims_B[2])
        self.fusion4 = CrossAttentionFusionModule(self.dims_A[3], self.dims_B[3])
        self.fusion5 = CrossAttentionFusionModule(self.dims_A[4], self.dims_B[4])

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.lka_attention = LKA_Attention3d(d_model=dims[4])

        # Decoder with attention gates
        self.Up5_A = up_conv_3d(self.dims_A[4], self.dims_A[3])
        self.Up5_B = up_conv_3d(self.dims_B[4], self.dims_B[3])
        self.Att5_A = AttentionGate3D(self.dims_A[3], self.dims_A[3], self.dims_A[2])
        self.Att5_B = AttentionGate3D(self.dims_B[3], self.dims_B[3], self.dims_B[2])
        self.Up_conv5_A = fusion_conv_3d(self.dims_A[3] * 2, self.dims_A[3])
        self.Up_conv5_B = fusion_conv_3d(self.dims_B[3] * 2, self.dims_B[3])
        self.decoder_fusion4 = CrossAttentionFusionModule(self.dims_A[3], self.dims_B[3])

        self.Up4_A = up_conv_3d(self.dims_A[3], self.dims_A[2])
        self.Up4_B = up_conv_3d(self.dims_B[3], self.dims_B[2])
        self.Att4_A = AttentionGate3D(self.dims_A[2], self.dims_A[2], self.dims_A[1])
        self.Att4_B = AttentionGate3D(self.dims_B[2], self.dims_B[2], self.dims_B[1])
        self.Up_conv4_A = fusion_conv_3d(self.dims_A[2] * 2, self.dims_A[2])
        self.Up_conv4_B = fusion_conv_3d(self.dims_B[2] * 2, self.dims_B[2])
        self.decoder_fusion3 = CrossAttentionFusionModule(self.dims_A[2], self.dims_B[2])

        self.Up3_A = up_conv_3d(self.dims_A[2], self.dims_A[1])
        self.Up3_B = up_conv_3d(self.dims_B[2], self.dims_B[1])
        self.Att3_A = AttentionGate3D(self.dims_A[1], self.dims_A[1], self.dims_A[0])
        self.Att3_B = AttentionGate3D(self.dims_B[1], self.dims_B[1], self.dims_B[0])
        self.Up_conv3_A = fusion_conv_3d(self.dims_A[1] * 2, self.dims_A[1])
        self.Up_conv3_B = fusion_conv_3d(self.dims_B[1] * 2, self.dims_B[1])
        self.decoder_fusion2 = CrossAttentionFusionModule(self.dims_A[1], self.dims_B[1])

        self.Up2_A = up_conv_3d(self.dims_A[1], self.dims_A[0])
        self.Up2_B = up_conv_3d(self.dims_B[1], self.dims_B[0])
        self.Att2_A = AttentionGate3D(self.dims_A[0], self.dims_A[0], self.dims_A[0] // 2)
        self.Att2_B = AttentionGate3D(self.dims_B[0], self.dims_B[0], self.dims_B[0] // 2)
        self.Up_conv2_A = fusion_conv_3d(self.dims_A[0] * 2, self.dims_A[0])
        self.Up_conv2_B = fusion_conv_3d(self.dims_B[0] * 2, self.dims_B[0])

        # Cascaded velocity heads
        self.flow1 = nn.Conv3d(dims[2], 3 * M, kernel_size=kernel_size, padding=kernel_size // 2)
        self.flow2 = nn.Conv3d(dims[1] + 3 * M, 3 * M, kernel_size=kernel_size, padding=kernel_size // 2)
        self.flow3 = nn.Conv3d(dims[0] + 3 * M, 3 * M, kernel_size=kernel_size, padding=kernel_size // 2)
        for conv in (self.flow1, self.flow2, self.flow3):
            nn.init.normal_(conv.weight, 0.0, 1e-5)
            nn.init.constant_(conv.bias, 0.0)

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stream_a = x[:, 0:1]
        stream_b = x[:, 1:]

        x1_a = self.encoder1_A(self.stem_A(stream_a))
        x1_b = self.encoder1_B(self.stem_B(stream_b))
        x2_a = self.encoder2_A(self.Maxpool(x1_a))
        x2_b = self.encoder2_B(self.Maxpool(x1_b))
        x3_a = self.encoder3_A(self.Maxpool(x2_a))
        x3_b = self.encoder3_B(self.Maxpool(x2_b))
        x4_a = self.encoder4_A(self.Maxpool(x3_a))
        x4_b = self.encoder4_B(self.Maxpool(x3_b))
        x5_a = self.encoder5_A(self.Maxpool(x4_a))
        x5_b = self.encoder5_B(self.Maxpool(x4_b))

        bottleneck = self.lka_attention(self.fusion5(x5_a, x5_b))
        d5_a_in, d5_b_in = torch.split(bottleneck, [self.dims_A[4], self.dims_B[4]], dim=1)

        d5_a = self.Up5_A(d5_a_in)
        d5_b = self.Up5_B(d5_b_in)
        x4_a_att = self.Att5_A(d5_a, x4_a)
        x4_b_att = self.Att5_B(d5_b, x4_b)
        d4_a = self.Up_conv5_A(torch.cat([x4_a_att, d5_a], dim=1))
        d4_b = self.Up_conv5_B(torch.cat([x4_b_att, d5_b], dim=1))
        d4_fused = self.decoder_fusion4(d4_a, d4_b)
        d4_a_out, d4_b_out = torch.split(d4_fused, [self.dims_A[3], self.dims_B[3]], dim=1)

        d4_a_up = self.Up4_A(d4_a_out)
        d4_b_up = self.Up4_B(d4_b_out)
        x3_a_att = self.Att4_A(d4_a_up, x3_a)
        x3_b_att = self.Att4_B(d4_b_up, x3_b)
        d3_a = self.Up_conv4_A(torch.cat([x3_a_att, d4_a_up], dim=1))
        d3_b = self.Up_conv4_B(torch.cat([x3_b_att, d4_b_up], dim=1))
        d3_fused = self.decoder_fusion3(d3_a, d3_b)
        vf1_small = self.flow1(d3_fused)

        d3_a_out, d3_b_out = torch.split(d3_fused, [self.dims_A[2], self.dims_B[2]], dim=1)
        d3_a_up = self.Up3_A(d3_a_out)
        d3_b_up = self.Up3_B(d3_b_out)
        x2_a_att = self.Att3_A(d3_a_up, x2_a)
        x2_b_att = self.Att3_B(d3_b_up, x2_b)
        d2_a = self.Up_conv3_A(torch.cat([x2_a_att, d3_a_up], dim=1))
        d2_b = self.Up_conv3_B(torch.cat([x2_b_att, d3_b_up], dim=1))
        d2_fused = self.decoder_fusion2(d2_a, d2_b)

        vf1_medium = self.up(vf1_small)
        vf2_medium = vf1_medium + self.flow2(torch.cat([d2_fused, vf1_medium], dim=1))

        d2_a_out, d2_b_out = torch.split(d2_fused, [self.dims_A[1], self.dims_B[1]], dim=1)
        d2_a_up = self.Up2_A(d2_a_out)
        d2_b_up = self.Up2_B(d2_b_out)
        x1_a_att = self.Att2_A(d2_a_up, x1_a)
        x1_b_att = self.Att2_B(d2_b_up, x1_b)
        d1_a = self.Up_conv2_A(torch.cat([x1_a_att, d2_a_up], dim=1))
        d1_b = self.Up_conv2_B(torch.cat([x1_b_att, d2_b_up], dim=1))
        d1_fused = torch.cat([d1_a, d1_b], dim=1)

        vf2_full = self.up(vf2_medium)
        vf3_full = vf2_full + self.flow3(torch.cat([d1_fused, vf2_full], dim=1))
        vf1_full = self.up(vf1_medium)

        vf1 = vf1_full.reshape(self.M, 3, *vf1_full.shape[2:])
        vf2 = vf2_full.reshape(self.M, 3, *vf2_full.shape[2:])
        vf3 = vf3_full.reshape(self.M, 3, *vf3_full.shape[2:])

        if self.R == 3:
            return torch.cat([vf1, vf2, vf3], dim=0)
        if self.R == 2:
            return torch.cat([vf2, vf3], dim=0)
        if self.R == 1:
            return vf3
        return torch.cat([vf1, vf2, vf3], dim=0)

# ------------------------- Temporal Attention -------------------------
class AttentionNet(nn.Module):
    """Temporal attention predictor for weighting stationary velocity fields."""

    def __init__(self, hidden_channels: int = 16, M: int = 2, R: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_channels * 4)
        self.fc2 = nn.Linear(hidden_channels * 4, hidden_channels * 8)
        self.fc3 = nn.Linear(hidden_channels * 8, hidden_channels * 8)
        self.fc4 = nn.Linear(hidden_channels * 8, hidden_channels * 4)
        self.fc5 = nn.Linear(hidden_channels * 4, M * R)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(self.fc1(t), 0.2)
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
    ) -> None:
        super().__init__()
        self.M = int(M)
        self.R = int(R)
        self.vf_net = CMUNeXt_VFNet_Final(C_in, C_hid, depths, kernels, M=self.M, R=self.R).to(device)
        self.att_net = AttentionNet(hidden_channels=16, M=self.M, R=self.R).to(device)

        self.h = float(step_size)
        self.num_steps = max(1, int(round(1.0 / self.h)))
        self.timesteps = torch.arange(self.num_steps, device=device)[:, None] * self.h
        self.scale = torch.as_tensor(inshape, device=device, dtype=torch.float32)[None, None, :] - 1.0

    def forward(self, vertices: torch.Tensor, volumes: torch.Tensor, return_extras: bool = False):
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

        attention = self.att_net(self.timesteps)[..., None, None]

        current_vertices = vertices
        trajectory_stats = [] if return_extras else None
        for step in range(self.num_steps):
            weights = attention[step]

            def sample_field(sample_vertices: torch.Tensor) -> torch.Tensor:
                raw = self.interpolate(sample_vertices, svfs_all)
                return (weights.view(-1, 1, 1) * raw).sum(0, keepdim=True)

            k1 = sample_field(current_vertices)
            k2 = sample_field(current_vertices + self.h * 0.5 * k1)
            k3 = sample_field(current_vertices + self.h * 0.5 * k2)
            k4 = sample_field(current_vertices + self.h * k3)
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
            per_level[f"vf{level + 1}"] = svfs_all[offset : offset + self.M].detach()
            offset += self.M

        extras = {
            "svfs_all": svfs_all.detach(),
            "per_level": per_level,
            "att_weights": attention.squeeze(-1).squeeze(-1).detach(),
            "traj_stats": trajectory_stats,
        }
        return current_vertices, extras

    def interpolate(self, vertices: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        coords = 2.0 * vertices / self.scale - 1.0
        coords = coords.repeat(fields.shape[0], 1, 1)
        coords = coords[:, :, None, None].flip(-1)
        sampled = F.grid_sample(fields, coords, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled[..., 0, 0].permute(0, 2, 1)
