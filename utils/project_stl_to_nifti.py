import argparse
import numpy as np
import nibabel as nib
import trimesh

def project_stl_to_nifti():
    # 参数解析（带默认值）
    parser = argparse.ArgumentParser(
        description='Project STL surface onto NIfTI space',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--stl', type=str,
        default='/path/to/your/file.stl',
        help='Input STL file path')
    parser.add_argument('--nii', type=str,
        default='/path/to/your/file.nii.gz',
        help='Reference NIfTI file path')
    parser.add_argument('--output', type=str,
        default=/path/to/your/file.nii.gz',
        help='Output NIfTI file path')
    args = parser.parse_args()

    # 加载参考NIfTI文件
    ref_img = nib.load(args.nii)
    ref_affine = ref_img.affine
    header = ref_img.header
    ref_shape = ref_img.shape

    # 初始化二进制掩模矩阵
    output_data = np.zeros(ref_shape, dtype=np.uint8)

    # 使用Trimesh加载STL文件
    mesh = trimesh.load(args.stl)
    vertices = mesh.vertices  # 获取顶点坐标矩阵[N×3]

    # 坐标转换：世界坐标 -> 体素空间
    inv_affine = np.linalg.inv(ref_affine)
    vox_coords = nib.affines.apply_affine(inv_affine, vertices)

    # 过滤有效体素索引
    vox_indices = np.round(vox_coords).astype(int)
    valid_mask = (
        (vox_indices[:,0] >= 0) & (vox_indices[:,0] < ref_shape[0]) &
        (vox_indices[:,1] >= 0) & (vox_indices[:,1] < ref_shape[1]) &
        (vox_indices[:,2] >= 0) & (vox_indices[:,2] < ref_shape[2])
    )
    valid_indices = vox_indices[valid_mask]

    # 标记被STL顶点覆盖的体素
    unique_indices = np.unique(valid_indices, axis=0)
    output_data[tuple(unique_indices.T)] = 1  # 高效索引赋值

    # 保存结果（保持原始空间属性）
    result_img = nib.Nifti1Image(output_data, ref_affine, header=header)
    nib.save(result_img, args.output)

if __name__ == '__main__':
    project_stl_to_nifti()
