"""Normalization and attention utilities for DDSurfer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


DEFAULT_REFINEMENT = {
    "norm_type": "group",
    "use_se": False,
    "residual_gates": True,
    "geometry_aware_lka": True,
    "conditioned_attention": True,
    "rk4_stage_time": True,
    "encoder_fusion": True,
    "cross_gate_policy": "conservative",
}


def resolve_refinement(overrides=None):
    result = DEFAULT_REFINEMENT.copy()
    unknown = set(overrides or {}) - set(result)
    if unknown:
        raise ValueError(f"Unknown refinement options: {sorted(unknown)}")
    result.update(overrides or {})
    if result["norm_type"] not in ("batch", "group", "instance"):
        raise ValueError("norm_type must be batch, group or instance")
    if result["cross_gate_policy"] not in ("standard", "conservative"):
        raise ValueError("cross_gate_policy must be standard or conservative")
    if result["cross_gate_policy"] == "conservative" and not result["residual_gates"]:
        raise ValueError("Conservative cross-stream modulation requires residual_gates=True")
    for key, value in result.items():
        if key not in ("norm_type", "cross_gate_policy") and not isinstance(value, bool):
            raise ValueError(f"{key} must be a bool")
    return result


def cross_gate_options(policy, stage):
    if policy == "standard":
        return {}
    if policy != "conservative" or stage not in range(1, 6):
        raise ValueError("Expected a known cross-stream policy and stage in 1..5")
    # Preserve shallow stream-specific features; allow more interaction at coarse scales.
    initial_gains = (0.01, 0.02, 0.03, 0.04, 0.05)
    maximum_gains = (0.03, 0.05, 0.08, 0.10, 0.15)
    return {"reduction": 8, "min_hidden": 2, "max_hidden": 32,
            "gain_init": initial_gains[stage - 1], "max_gain": maximum_gains[stage - 1]}


def make_norm(channels, norm_type="group", eps=1e-5):
    if norm_type == "batch":
        return nn.BatchNorm3d(channels, eps=eps)
    if norm_type == "instance":
        return nn.InstanceNorm3d(channels, eps=eps, affine=True, track_running_stats=False)
    if norm_type != "group":
        raise ValueError(f"Unknown norm: {norm_type}")
    # Dual-stream widths need not divide 8; avoid single-channel groups when possible.
    candidates = [g for g in range(1, min(8, channels) + 1)
                  if channels % g == 0 and channels // g >= 4]
    return nn.GroupNorm(max(candidates or [1]), channels, eps=eps)


def replace_batchnorm(module, norm_type):
    if norm_type == "batch":
        return
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm3d):
            replacement = make_norm(child.num_features, norm_type, child.eps)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            replace_batchnorm(child, norm_type)


class ResidualGate(nn.Module):
    """Centered, bounded modulation with an explicit identity path."""

    def __init__(self, gain_init=0.1, max_gain=0.25):
        super().__init__()
        if not 0 <= gain_init < max_gain < 1:
            raise ValueError("Expected 0 <= gain_init < max_gain < 1")
        self.max_gain = float(max_gain)
        self.raw_gain = nn.Parameter(torch.tensor(math.atanh(gain_init / max_gain)))
        self.collect_diagnostics = False
        self.last_gate_stats = None

    def forward(self, features, weights):
        gain = self.max_gain * self.raw_gain.tanh()
        multiplier = 1.0 + gain * (2.0 * weights - 1.0)
        if self.collect_diagnostics:
            with torch.no_grad():
                w = weights.float()
                self.last_gate_stats = {
                    "mean": float(w.mean()), "std": float(w.std(unbiased=False)),
                    "saturated_fraction": float(((w < 0.05) | (w > 0.95)).float().mean()),
                    "gain": float(gain), "multiplier_min": float(multiplier.min()),
                    "multiplier_max": float(multiplier.max()), "gain_limit": self.max_gain,
                }
        return features * multiplier


class ResidualSkipGate3D(nn.Module):
    def __init__(self, decoder_channels, encoder_channels, inter_channels, norm_type="group"):
        super().__init__()
        inter_channels = max(1, inter_channels)
        self.W_g = nn.Sequential(nn.Conv3d(decoder_channels, inter_channels, 1),
                                 make_norm(inter_channels, norm_type))
        self.W_x = nn.Sequential(nn.Conv3d(encoder_channels, inter_channels, 1),
                                 make_norm(inter_channels, norm_type))
        # Do not normalize scalar mask logits: preserve their calibrated magnitude.
        self.psi = nn.Sequential(nn.Conv3d(inter_channels, 1, 1), nn.Sigmoid())
        nn.init.zeros_(self.psi[0].weight)
        nn.init.zeros_(self.psi[0].bias)
        self.relu = nn.ReLU(inplace=True)
        self.modulation = ResidualGate()

    def forward(self, decoder_features, encoder_features):
        logits = self.relu(self.W_g(decoder_features) + self.W_x(encoder_features))
        return self.modulation(encoder_features, self.psi(logits))


def fitting_odd_kernel(shape, maximum):
    result = []
    for extent in shape:
        if extent < 1:
            raise ValueError("Feature-map extents must be positive")
        k = min(int(extent), maximum)
        result.append(k if k % 2 else k - 1)
    return tuple(result)


class GeometryAwareLKA3D(nn.Module):
    """VAN-style DW/DW/pointwise attention adapted to a small 3D bottleneck."""

    def __init__(self, channels, feature_shape, norm_type="group"):
        super().__init__()
        self.feature_shape = tuple(feature_shape)
        local_kernel = fitting_odd_kernel(feature_shape, 3)
        spatial_kernel = fitting_odd_kernel(feature_shape, 7)
        self.norm = make_norm(channels, norm_type)
        self.proj_1 = nn.Conv3d(channels, channels, 1)
        self.activation = nn.GELU()
        self.conv0 = nn.Conv3d(channels, channels, local_kernel,
                               padding=tuple(k // 2 for k in local_kernel), groups=channels)
        self.conv_spatial = nn.Conv3d(channels, channels, spatial_kernel,
                                      padding=tuple(k // 2 for k in spatial_kernel), groups=channels)
        self.conv1 = nn.Conv3d(channels, channels, 1)
        self.proj_2 = nn.Conv3d(channels, channels, 1)
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), 1e-2))

    def forward(self, features):
        if tuple(features.shape[2:]) != self.feature_shape:
            raise ValueError("LKA feature shape differs from the architecture's planned bottleneck")
        x = self.activation(self.proj_1(self.norm(features)))
        mask = self.conv1(self.conv_spatial(self.conv0(x)))
        return features + self.layer_scale * self.proj_2(x * mask)
