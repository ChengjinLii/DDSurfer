import os
import argparse
import SimpleITK as sitk


def get_image_properties(image_path):
    """
    获取 NIfTI 图像的 spacing、size、origin 和 direction。

    Args:
        image_path: 图像路径。
    Returns:
        A dictionary 包含 spacing, size, origin, direction。
    """
    image = sitk.ReadImage(image_path)
    properties = {
        "spacing": image.GetSpacing(),
        "size": image.GetSize(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
    }
    return properties


def resample_image_to_target_space(source_image_path, target_image_path, output_file_path):
    """
    将源图像重新采样到目标图像空间。

    Args:
        source_image_path: 源图像路径。
        target_image_path: 目标图像路径。
        output_file_path: 输出文件路径。
    """
    # 加载源图像和目标图像
    source_image = sitk.ReadImage(source_image_path)
    target_image = sitk.ReadImage(target_image_path)

    # 获取目标图像属性
    target_spacing = target_image.GetSpacing()
    target_size = target_image.GetSize()
    target_origin = target_image.GetOrigin()
    target_direction = target_image.GetDirection()

    print("Target Image Properties:")
    print(f"  Spacing: {target_spacing}")
    print(f"  Size: {target_size}")
    print(f"  Origin: {target_origin}")
    print(f"  Direction: {target_direction}")

    # 设置重采样滤波器
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetInterpolator(sitk.sitkLinear)  # 使用线性插值

    # 对源图像进行重采样
    resampled_image = resampler.Execute(source_image)

    # 保存重采样后的图像
    sitk.WriteImage(resampled_image, output_file_path)

    print(f"\nResampled image saved to: {output_file_path}")


if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description="Resample an image to the target space.")
    parser.add_argument(
        "--source_image_path", 
        type=str, 
        required=True, 
        help="Path to the source NIfTI image."
    )
    parser.add_argument(
        "--target_image_path", 
        type=str, 
        required=True, 
        help="Path to the target NIfTI image."
    )
    parser.add_argument(
        "--output_file_path", 
        type=str, 
        required=True, 
        help="Path to the output NIfTI image file."
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 重采样图像
    resample_image_to_target_space(
        source_image_path=args.source_image_path, 
        target_image_path=args.target_image_path, 
        output_file_path=args.output_file_path
    )
