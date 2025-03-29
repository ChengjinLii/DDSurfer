import nibabel as nib
import numpy as np
import trimesh
import os
import sys

# read_surface 函数 (包含LPS->RAS转换)
def read_surface(surface_path):
    print(f"尝试加载表面文件: {surface_path}")
    if not os.path.exists(surface_path):
        print(f"错误: 表面文件未找到于 {surface_path}")
        return None, None

    file_ext = os.path.splitext(surface_path)[1].lower()

    # --- 优先: FreeSurfer 原生格式 ---
    if file_ext in ['.pial', '.white', '.inflated', '.sphere', '.orig', '.smoothwm']:
        try:
            vertices, faces = nib.freesurfer.io.read_geometry(surface_path)
            print(f"使用 nibabel 加载了 {len(vertices)} 个顶点 (原生格式, 假定为 RAS)。")
            return vertices, faces
        except Exception as e:
            print(f"使用 nibabel 加载 FreeSurfer 表面 {surface_path} 出错: {e}")
            return None, None

    # --- 备选: STL 格式 (假定为 LPS, 将转换为 RAS) ---
    elif file_ext == '.stl':
        print("信息: 检测到 STL 文件。假定其顶点位于 LPS 坐标空间。")
        print("      将执行 LPS -> RAS 的转换 (x'=-x, y'=-y, z'=z)。")
        try:
            mesh_obj = trimesh.load_mesh(surface_path)
            if isinstance(mesh_obj, trimesh.Scene):
                if len(mesh_obj.geometry) == 1:
                    mesh_obj = list(mesh_obj.geometry.values())[0]
                else:
                    print(f"警告: STL 场景包含多个几何体。将它们合并。")
                    mesh_obj = trimesh.util.concatenate(
                        list(mesh_obj.geometry.values())
                    )
            if not isinstance(mesh_obj, trimesh.Trimesh):
                 raise TypeError(f"加载的对象不是 Trimesh 网格: {type(mesh_obj)}")

            vertices_lps = mesh_obj.vertices
            faces = mesh_obj.faces
            print(f"使用 trimesh 从 STL 加载了 {len(vertices_lps)} 个顶点 (LPS)。")

            print("      正在将顶点从 LPS 转换为 RAS...")
            vertices_ras = vertices_lps.copy()
            vertices_ras[:, 0] = -vertices_lps[:, 0] # X -> -X
            vertices_ras[:, 1] = -vertices_lps[:, 1] # Y -> -Y
            print(f"      转换完成。现在顶点为 RAS 坐标系。")
            return vertices_ras, faces
        except Exception as e:
            print(f"处理 STL 文件 {surface_path} (加载或转换) 出错: {e}")
            return None, None
    else:
        print(f"警告: 不支持的表面文件格式 '{file_ext}'。")
        return None, None


def create_label_excluding_value_simple(surface_file_path, mgz_file_path, output_label_path,
                                        value_to_exclude):
    """
    生成一个 FreeSurfer .label 文件，包含所有不对应于 MGZ 中特定标签值的表面顶点。
    使用最近邻采样。

    Args:
        surface_file_path (str): 输入表面文件的路径。
        mgz_file_path (str): 输入 MGZ 分割文件的路径。
        output_label_path (str): 保存输出 .label 文件的路径。
        value_to_exclude (int): 需要从标签文件中排除的 MGZ 标签值。
    """
    # --- 加载表面 (read_surface 会处理坐标转换) ---
    surface_vertices_ras, _ = read_surface(surface_file_path)
    if surface_vertices_ras is None:
        print("加载表面顶点失败。中止。")
        return False
    n_vertices = len(surface_vertices_ras)

    # --- 加载 MGZ ---
    print(f"加载 MGZ 分割文件: {mgz_file_path}")
    if not os.path.exists(mgz_file_path):
        print(f"错误: MGZ 文件未找到于 {mgz_file_path}")
        return False
    try:
        mgz_img = nib.load(mgz_file_path)
        mgz_data = mgz_img.get_fdata()
        mgz_affine_vox_to_ras = mgz_img.affine
        inv_mgz_affine_ras_to_vox = np.linalg.inv(mgz_affine_vox_to_ras)
        dims = mgz_data.shape
        print(f"MGZ 数据形状: {dims}")
    except Exception as e:
        print(f"加载 MGZ 文件 {mgz_file_path} 出错: {e}")
        return False

    # --- 将表面顶点 (已经是 RAS) 转换到体素坐标 ---
    print("转换表面顶点 (RAS) 到 MGZ 体素坐标...")
    vertices_ras_homogeneous = np.hstack((surface_vertices_ras, np.ones((n_vertices, 1))))
    voxel_coords_homogeneous = inv_mgz_affine_ras_to_vox @ vertices_ras_homogeneous.T
    voxel_coords = voxel_coords_homogeneous[:3, :].T

    # --- 采样体素标签 (最近邻) ---
    print("为所有顶点采样体素标签 (最近邻)...")
    voxel_indices = np.round(voxel_coords).astype(int)
    # 裁剪索引
    voxel_indices[:, 0] = np.clip(voxel_indices[:, 0], 0, dims[0] - 1)
    voxel_indices[:, 1] = np.clip(voxel_indices[:, 1], 0, dims[1] - 1)
    voxel_indices[:, 2] = np.clip(voxel_indices[:, 2], 0, dims[2] - 1)

    try:
        vertex_labels = mgz_data[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]]
    except IndexError as e:
         print(f"标签采样过程中出错: {e}")
         return False

    # --- 直接筛选：保留所有标签不等于 value_to_exclude 的顶点 ---
    print(f"筛选顶点: 保留标签不等于 {value_to_exclude} 的顶点...")
    keep_mask = (vertex_labels != value_to_exclude)
    kept_vertex_indices = np.where(keep_mask)[0]

    # --- 报告结果 ---
    num_kept_vertices = len(kept_vertex_indices)
    num_excluded = n_vertices - num_kept_vertices
    print(f"排除了 {num_excluded} 个顶点 (标签为 {value_to_exclude}), 保留了 {num_kept_vertices} 个顶点。")
    if num_kept_vertices == 0:
        print(f"警告: 没有顶点被保留下来。检查标签值 {value_to_exclude} 是否覆盖了整个表面。")
        print("  将写入一个空的 label 文件。")

    # --- 写入 FreeSurfer Label 文件 ---
    print(f"写入 label 文件到: {output_label_path}")
    kept_coords_ras = surface_vertices_ras[kept_vertex_indices]

    try:
        os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
        with open(output_label_path, 'w') as f:
            f.write("#!ascii label, generated by script\n")
            f.write(f"{num_kept_vertices}\n")
            for i in range(num_kept_vertices):
                vertex_index = kept_vertex_indices[i]
                coords = kept_coords_ras[i]
                f.write(f"{vertex_index}  {coords[0]:.6f}  {coords[1]:.6f}  {coords[2]:.6f}  0.000000\n")
        print("Label 文件成功创建。")
        return True

    except Exception as e:
        print(f"写入 label 文件 {output_label_path} 出错: {e}")
        return False


# --- 配置 ---
subject_dir = "/path"
mgz_segmentation_file = os.path.join(subject_dir, "100307-DDSurfer-wmparc-Reg.mgz")
output_dir = os.path.join(subject_dir, "label")
# 使用反映简单逻辑的文件名
output_label_file = os.path.join(output_dir, "lh.cortex.label")

# 要排除的标签值
label_to_exclude = 192

# --- 寻找表面文件 ---
fs_surf_dir = os.path.join(subject_dir, "surf")
native_surface_file_pial = os.path.join(fs_surf_dir, "lh.pial")
native_surface_file_white = os.path.join(fs_surf_dir, "lh.white")
surface_to_use = None
# 优先使用 pial 表面
if os.path.exists(native_surface_file_pial):
    surface_to_use = native_surface_file_pial
    print(f"使用原生表面 (RAS): {surface_to_use}")
elif os.path.exists(native_surface_file_white):
    surface_to_use = native_surface_file_white
    print(f"使用原生表面 (RAS): {surface_to_use}")
else:
    print("未找到原生 FreeSurfer 表面。")
    stl_surface_file = os.path.join(subject_dir, "surf", "100307_predicted_wm_surface_left.stl")
    if os.path.exists(stl_surface_file):
         print(f"回退到使用 STL 文件 (LPS): {stl_surface_file}")
         surface_to_use = stl_surface_file
    else:
         print(f"错误: 既未找到原生表面，也未找到指定的 STL 文件。")
         surface_to_use = None

# --- 运行 Label 创建 ---
if surface_to_use:
    print(f"\n--- 创建 Label 文件: 直接排除标签值为 {label_to_exclude} (简单最近邻) ---")
    success = create_label_excluding_value_simple( # 调用新函数名
        surface_file_path=surface_to_use,
        mgz_file_path=mgz_segmentation_file,
        output_label_path=output_label_file,
        value_to_exclude=label_to_exclude
    )
    if success:
        print("\n处理成功完成。")
    else:
        print("\n处理过程中出现错误。")
        sys.exit(1)
else:
    print("\n没有有效的表面文件，无法继续。")
    sys.exit(1)
