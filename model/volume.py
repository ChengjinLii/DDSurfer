import torch
import torch.nn as nn
import torch.nn.functional as F

"""Springer2023
最近，深度学习已被证明在多视角立体视觉 (MVS) 中表现出色。然而，基于深度学习的 MVS 方法很难平衡其效率和有效性。为此，我们提出了 DSC-MVSNet，这是一种新颖的粗到精和端到端框架，用于在 MVS 中更高效、更准确地估计深度。
具体来说，我们提出了一种注意力感知的 3D UNet-shape 网络，它首先使用深度可分离卷积进行成本体积正则化。该机制通过将成本体积上的普通卷积转换为深度卷积和点卷积，实现了信息的有效聚合，并显著减少了模型参数和计算量。
此外，提出了一个 3D-Attention 模块来缓解成本体积正则化中的特征不匹配问题，并在三个维度（即通道、空间和深度）中聚合成本体积的重要信息。
此外，我们提出了一个有效的特征传输模块，将低分辨率 (LR) 深度图上采样为高分辨率 (HR) 深度图，以实现更高的精度。
通过在两个基准数据集（即 DTU 和 Tanks & Temples）上进行大量实验，我们证明了我们的模型的参数显著减少到最先进的模型.


VolumeRegularizer模块起到的作用体现在以下几个方面：
代价体积正则化：通过使用深度可分离卷积（DSC）来对代价体积进行有效的正则化。正则化可以帮助平滑和优化代价体积，从而提高深度估计的准确性。
参数高效性：通过使用深度可分离卷积，VolumeRegularizer 模块显著减少了模型的参数量和计算开销，同时仍然能够保持性能。
注意力机制：该模块集成了3D注意力机制，用于缓解代价体积正则化过程中可能出现的特征不匹配问题。注意力机制能够在通道、空间和深度维度上聚合代价体积中的重要信息，提升特征的表达能力。
"""

class SeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(SeparableConv3D, self).__init__()

        # depthwise conv (dw)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm3d(in_channels)

        # pointswise conv (pw)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x


class SeparableUpConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(SeparableUpConv3D, self).__init__()

        # depthwise deconv (dedw)
        self.conv1 = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, output_padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm3d(in_channels)

        # pointwise deconv (depw)
        self.conv2 = nn.ConvTranspose3d(in_channels,out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # MLP
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        channel_att_raw = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, kernel_size=(x.size(2), x.size(3), x.size(4)),stride=(x.size(2), x.size(3), x.size(4))) # [B, C, 1, 1, 1]
                channel_att_raw = self.mlp(avg_pool) # [B, C]
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, kernel_size=(x.size(2), x.size(3), x.size(4)),stride=(x.size(2), x.size(3), x.size(4))) # [B, C, 1, 1, 1]
                channel_att_raw = self.mlp(max_pool) # [B, C]

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid(channel_att_sum) # [B, C]
        scale = scale.unsqueeze(2).unsqueeze(3).unsqueeze(3).expand_as(x) # [B, C, X.D, X.H, X.W]
        return x * scale

# fixme: Feature Extraction for Spatial Attention
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# fixme: Channel Max Pooling
class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

# fixme: Spatial Depth Attention
class SpatialDepthGate(nn.Module):
    def __init__(self):
        super(SpatialDepthGate, self).__init__()
        kernel_size = 7
        self.channel_pool = ChannelPool()
        self.channel_conv = BasicConv(2, 1, kernel_size=(1, kernel_size, kernel_size), stride=1, padding=(0, (kernel_size-1) // 2, (kernel_size-1) // 2), relu=False)  # 后面为了减少参数量可以考虑将它换为可分离卷积
        self.depth_conv = BasicConv(1, 1, kernel_size=(kernel_size, 1, 1), stride=1, padding=((kernel_size-1) // 2, 0, 0), relu=False)
        self.overall_conv = BasicConv(1, 1, kernel_size=(kernel_size,kernel_size,kernel_size), stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x): # x [B, C. X.D, X.H, X.W]
        compress = self.channel_pool(x) # [B, 2, X.D, X.H, X.W]  [B, 32, 12, 16,20]
        compress = self.channel_conv(compress) # [B, 1, 12 ,16, 20]
        compress = self.depth_conv(compress) # [B, 1, 12 ,16, 20]
        compress = self.overall_conv(compress) # [1, 1, 12, 16, 20]
        scale = F.sigmoid(compress) # [1, 1, 12, 16, 20]
        return x * scale

# fixme: 3D Attention Module
class VolumeAttention3D(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_type=['avg', 'max'], no_spatial_depth=False):
        super(VolumeAttention3D, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_type)
        self.no_spatial_depth = no_spatial_depth
        if not no_spatial_depth:
            self.SpatialDepthGate = SpatialDepthGate()

    def forward(self, x):
        x = self.ChannelGate(x)
        if not self.no_spatial_depth:
            x = self.SpatialDepthGate(x)
        return x

class VolumeRegularizer(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(VolumeRegularizer, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.output_channels = base_channels * 8

        # Basic Conv
        self.conv0_1 = SeparableConv3D(in_channels, base_channels, stride=1)

        self.conv1_1 = SeparableConv3D(base_channels * 2, base_channels * 2, stride=1)
        self.conv2_1 = SeparableConv3D(base_channels * 4, base_channels * 4, stride=1)
        self.conv3_1 = SeparableConv3D(base_channels * 8, base_channels * 8, stride=1)

        # Downsample
        self.conv1_0 = SeparableConv3D(in_channels, base_channels * 2, stride=2)
        self.conv2_0 = SeparableConv3D(base_channels * 2, base_channels * 4, stride=2)
        self.conv3_0 = SeparableConv3D(base_channels * 4, base_channels * 8, stride=2)

        # Upsample
        self.conv4_0 = SeparableUpConv3D(base_channels * 8, base_channels * 4, stride=2)
        self.conv5_0 = SeparableUpConv3D(base_channels * 4, base_channels * 2, stride=2)
        self.conv6_0 = SeparableUpConv3D(base_channels * 2, base_channels, stride=2)

        # VolumeAttention3D
        self.attention_block_1 = VolumeAttention3D(self.base_channels * 4, reduction_ratio=8, pool_type=['avg', 'max'], no_spatial_depth=False)
        self.attention_block_2 = VolumeAttention3D(self.base_channels * 2, reduction_ratio=8, pool_type=['avg', 'max'], no_spatial_depth=False)
        self.attention_block_3 = VolumeAttention3D(self.base_channels, reduction_ratio=8, pool_type=['avg', 'max'], no_spatial_depth=False)

    def forward(self, x):
        in_channels = x.shape[1]

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

        conv6_2 = nn.Conv3d(self.base_channels, in_channels, 3, padding=1, bias=False).to(x.device)
        output = conv6_2(self.attention_block_3(conv6_0 + conv0_1))

        return output


if __name__ == '__main__':
    #  [batch_size, channels, depth, height, width]
    input = torch.rand(1, 32, 48, 64, 80)  # [B, C, D, H, W]

    block = VolumeRegularizer(in_channels=32, base_channels=8)

    output = block(input)

    print(f"Input size: {input.size()}")
    print(f"Output size: {output.size()}")
