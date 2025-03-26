import torch
import torch.nn as nn

"""
3D CMUNEXT Network
------------------
This is a 3D version of the CMUNEXT network, which is a lightweight network for medical image segmentation. 
------------------
The network is based on the paper:
@INPROCEEDINGS{10635609,
  author={Tang, Fenghe and Ding, Jianrui and Quan, Quan and Wang, Lingtao and Ning, Chunping and Zhou, S. Kevin},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)}, 
  title={CMUNEXT: An Efficient Medical Image Segmentation Network Based on Large Kernel and Skip Fusion}, 
  year={2024},
  volume={},
  number={},
  pages={1-5},
  keywords={Location awareness;Image segmentation;Convolution;Computer architecture;Transformers;Computational efficiency;Data mining;Medical image segmentation;Lightweight network;Large Kernel;Skip-fusion},
  doi={10.1109/ISBI56570.2024.10635609}}
"""

class conv_block_3d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Residual3d(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class CMUNeXtBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock3D, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual3d(nn.Sequential(
                    # depth wise
                    nn.Conv3d(ch_in, ch_in, kernel_size=(k, k, k), groups=ch_in, padding=(k//2, k//2, k//2)),
                    nn.GELU(),
                    nn.BatchNorm3d(ch_in)
                )),
                nn.Conv3d(ch_in, ch_in*4, kernel_size=(1, 1, 1)),
                nn.GELU(),
                nn.BatchNorm3d(ch_in*4),
                nn.Conv3d(ch_in*4, ch_in, kernel_size=(1, 1, 1)),
                nn.GELU(),
                nn.BatchNorm3d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block_3d(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class up_conv_3d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_3d, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class fusion_conv_3d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusion_conv_3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm3d(ch_in),
            nn.Conv3d(ch_in, ch_out * 4, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm3d(ch_out * 4),
            nn.Conv3d(ch_out * 4, ch_out, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm3d(ch_out)
        )

    def forward(self, x):
        return self.conv(x)

class CMUNeXt3D(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, 
                dims=[16, 32, 128, 160, 256], 
                depths=[1, 1, 1, 3, 1], 
                kernels=[3, 3, 7, 7, 7]):
        super(CMUNeXt3D, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.stem = conv_block_3d(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock3D(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock3D(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock3D(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock3D(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock3D(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # Decoder
        self.Up5 = up_conv_3d(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv_3d(ch_in=dims[3]*2, ch_out=dims[3])
        self.Up4 = up_conv_3d(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv_3d(ch_in=dims[2]*2, ch_out=dims[2])
        self.Up3 = up_conv_3d(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv_3d(ch_in=dims[1]*2, ch_out=dims[1])
        self.Up2 = up_conv_3d(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv_3d(ch_in=dims[0]*2, ch_out=dims[0])
        self.Conv_1x1x1 = nn.Conv3d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoding
        x1 = self.stem(x)
        print(f"x1 = self.stem(x): {x1.shape}")
        x1 = self.encoder1(x1)
        print(f"x1 = self.encoder1(x1): {x1.shape}")
        x2 = self.Maxpool(x1)
        print(f"x2 = self.Maxpool(x1): {x2.shape}")
        x2 = self.encoder2(x2)
        print(f"x2 = self.encoder2(x2): {x2.shape}")
        x3 = self.Maxpool(x2)
        print(f"x3 = self.Maxpool(x2): {x3.shape}")
        x3 = self.encoder3(x3)
        print(f"x3 = self.encoder3(x3): {x3.shape}")
        x4 = self.Maxpool(x3)
        print(f"x4 = self.Maxpool(x3): {x4.shape}")
        x4 = self.encoder4(x4)
        print(f"x4 = self.encoder4(x4): {x4.shape}")
        
        x5 = self.Maxpool(x4)
        print(f"x5 = self.Maxpool(x4): {x5.shape}")
        x5 = self.encoder5(x5)
        print(f"x5 = self.encoder5(x5): {x5.shape}")

        # Decoding
        d5 = self.Up5(x5)
        print(f"d5 = self.Up5(x5): {d5.shape}")
        d5 = torch.cat((x4, d5), dim=1)
        print(f"d5 = torch.cat((x4, d5), dim=1): {d5.shape}")
        d5 = self.Up_conv5(d5)
        print(f"d5 = self.Up_conv5(d5): {d5.shape}")

        d4 = self.Up4(d5)
        print(f"d4 = self.Up4(d5): {d4.shape}")
        d4 = torch.cat((x3, d4), dim=1)
        print(f"d4 = torch.cat((x3, d4), dim=1): {d4.shape}")
        d4 = self.Up_conv4(d4)
        print(f"d4 = self.Up_conv4(d4): {d4.shape}")

        d3 = self.Up3(d4)
        print(f"d3 = self.Up3(d4): {d3.shape}")
        d3 = torch.cat((x2, d3), dim=1)
        print(f"d3 = torch.cat((x2, d3), dim=1): {d3.shape}")
        d3 = self.Up_conv3(d3)
        print(f"d3 = self.Up_conv3(d3): {d3.shape}")

        d2 = self.Up2(d3)
        print(f"d2 = self.Up2(d3): {d2.shape}")
        d2 = torch.cat((x1, d2), dim=1)
        print(f"d2 = torch.cat((x1, d2), dim=1): {d2.shape}")
        d2 = self.Up_conv2(d2)
        print(f"d2 = self.Up_conv2(d2): {d2.shape}")

        d1 = self.Conv_1x1x1(d2)
        print(f"d1 = self.Conv_1x1x1(d2): {d1.shape}")

        return d1


if __name__ == '__main__':
    # 配置参数
    batch_size = 1
    in_channels = 7       # 输入通道数
    num_classes = 2        # 输出类别数
    input_shape = (112, 224, 176)  # (depth, height, width)

    # 初始化模型
    model = CMUNeXt3D(
        input_channel=in_channels,
        num_classes=num_classes,
        dims=[16, 32, 64, 128, 256],  # 调整通道数适应输入尺寸
        depths=[1, 1, 1, 2, 1],
        kernels=[3, 3, 5, 5, 5]       # 减小高层kernel尺寸
    )

    # 生成测试输入 (batch, channel, depth, height, width)
    test_input = torch.randn(batch_size, in_channels, *input_shape)
    print(f"输入尺寸: {test_input.shape}")

    # 前向传播测试
    try:
        output = model(test_input)
        print(f"\n输出尺寸: {output.shape}")
        print("尺寸验证结果: 运行成功!")
        
        # 验证输入输出尺寸匹配
        assert output.shape[2:] == test_input.shape[2:], "空间尺寸不匹配!"
        assert output.shape[1] == num_classes, "输出通道数错误!"
        
    except Exception as e:
        print(f"\n错误信息: {str(e)}")
        print("尺寸验证结果: 运行失败!")

    # 各层尺寸调试信息
    print("\n各层尺寸变化追踪：")
    
    def print_shape(module, input, output):
        print(f"{module.__class__.__name__:20} | 输入: {tuple(input[0].shape)} → 输出: {tuple(output.shape)}")

    # 注册hook跟踪关键层
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (conv_block_3d, CMUNeXtBlock3D, fusion_conv_3d, up_conv_3d)):
            hook = layer.register_forward_hook(print_shape)
            hooks.append(hook)

    # 再次运行带跟踪的前向传播
    _ = model(test_input)
    
    # 移除hook
    for hook in hooks:
        hook.remove()
