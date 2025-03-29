import argparse
import nibabel as nib
import numpy as np

def skull_stripping(input_path, mask_path, output_path):
    # 加载输入文件和掩膜文件
    input_img = nib.load(input_path)
    mask_img = nib.load(mask_path)
    
    # 获取数据数组
    input_data = input_img.get_fdata()
    mask_data = mask_img.get_fdata()
    
    # 进行颅骨去除（使用掩膜）
    stripped_data = input_data * mask_data
    
    # 创建新的图像对象并保存
    stripped_img = nib.Nifti1Image(stripped_data, input_img.affine)
    nib.save(stripped_img, output_path)
    print(f"Skull-stripped image saved to {output_path}")

if __name__ == "__main__":
    # 设置argparse
    parser = argparse.ArgumentParser(description="Skull stripping using a mask")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input NIfTI file")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to mask NIfTI file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output NIfTI file")
    
    # 解析参数
    args = parser.parse_args()
    
    # 执行颅骨去除
    skull_stripping(args.input_path, args.mask_path, args.output_path)
