import argparse
import os
import numpy as np
import trimesh
import SimpleITK as sitk
import time

NUM_DEBUG_VERTICES = 5 # 打印前 N 个顶点的坐标用于调试
PRINT_TRANSFORM_DETAILS = True # 是否打印加载的变换信息

# --- 函数定义 ---

def transform_mesh_with_tfm(input_stl_path, output_stl_path, tfm_transform_path):
    """
    加载 STL 网格 (LPS)，转换为 RAS，使用 SimpleITK 加载 TFM 变换文件并应用，
    保存结果 (RAS)。
    """
    print("\n--- [开始 STL 变换 (使用 TFM)] ---")
    print(f"输入 STL 文件:      {input_stl_path}")
    print(f"使用 TFM 变换文件: {tfm_transform_path}")
    print(f"输出 STL 文件:      {output_stl_path}")

    # --- 检查输入文件 ---
    if not os.path.exists(input_stl_path):
        print(f"错误：输入 STL 文件未找到: {input_stl_path}")
        return False
    if not os.path.exists(tfm_transform_path):
        print(f"错误：TFM 变换文件未找到: {tfm_transform_path}")
        return False

    try:
        # 1. 加载 TFM 变换 (使用 SimpleITK)
        print("\n[步骤 1] 使用 SimpleITK 加载 TFM 变换...")
        try:
            transform = sitk.ReadTransform(tfm_transform_path)
            print("SimpleITK 加载 TFM 变换成功。")
            if PRINT_TRANSFORM_DETAILS:
                print(f"  - 加载的变换类型: {transform.GetName()}")
                try:
                    fixed_params = transform.GetFixedParameters()
                    print(f"  - 固定参数 (旋转中心): {fixed_params}")
                except Exception as e_fp:
                    print(f"  - 无法获取固定参数: {e_fp}")
                try:
                    params = transform.GetParameters()
                    print(f"  - 变换参数: {params}")
                except Exception as e_p:
                    print(f"  - 无法获取变换参数: {e_p}")
        except Exception as e_sitk_read:
            print(f"错误：使用 SimpleITK 读取 TFM 文件 '{tfm_transform_path}' 时失败: {e_sitk_read}")
            return False

        # 2. 加载 STL 网格 (假设为 LPS 方向)
        print("\n[步骤 2] 加载输入 STL 网格...")
        mesh_lps = trimesh.load(input_stl_path, process=False)
        num_vertices_total = len(mesh_lps.vertices)
        num_faces_total = len(mesh_lps.faces)
        print(f"STL 加载成功: {num_vertices_total} 个顶点, {num_faces_total} 个面。")
        vertices_lps = mesh_lps.vertices
        if NUM_DEBUG_VERTICES > 0:
            print(f"  - 前 {NUM_DEBUG_VERTICES} 个顶点的 LPS 坐标:\n{vertices_lps[:NUM_DEBUG_VERTICES]}")

        # 3. 坐标系转换: LPS -> RAS
        print("\n[步骤 3] 将顶点从 LPS 转换为 RAS...")
        vertices_ras = vertices_lps * np.array([-1, -1, 1])
        print("LPS -> RAS 转换完成。")
        if NUM_DEBUG_VERTICES > 0:
            print(f"  - 前 {NUM_DEBUG_VERTICES} 个顶点的 RAS 坐标:\n{vertices_ras[:NUM_DEBUG_VERTICES]}")

        # 4. 应用逆变换 (使用 SimpleITK 的 TransformPoint 方法)
        print("\n[步骤 4] 使用 SimpleITK 应用 TFM 变换到每个 RAS 顶点...")
        start_time = time.time()
        # 使用列表推导式稍微提高效率
        transformed_vertices_ras = np.array([
            list(transform.TransformPoint(tuple(float(c) for c in v)))
            for v in vertices_ras
        ])
        elapsed_time = time.time() - start_time
        print(f"SimpleITK 变换应用完成，耗时: {elapsed_time:.2f} 秒。")
        if NUM_DEBUG_VERTICES > 0:
            print(f"  - 前 {NUM_DEBUG_VERTICES} 个最终变换后的 3D RAS 坐标:\n{transformed_vertices_ras[:NUM_DEBUG_VERTICES]}")

        # 5. 创建新的 Trimesh 对象
        print("\n[步骤 5] 创建新的 Trimesh 网格对象...")
        output_mesh = trimesh.Trimesh(vertices=transformed_vertices_ras,
                                      faces=mesh_lps.faces, # 使用原始的面连接信息
                                      process=False)
        print("新网格对象创建成功。")

        # 6. 保存变换后的网格 (位于原始空间的 RAS 方向)
        print("\n[步骤 6] 保存变换后的 STL 文件...")
        output_dir = os.path.dirname(output_stl_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"  - 输出目录不存在，创建: {output_dir}")
            os.makedirs(output_dir)
        output_mesh.export(output_stl_path)
        print(f"变换后的 STL 文件已保存到: {output_stl_path}")
        print("\n--- [STL 变换成功完成] ---")
        return True

    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细的错误堆栈
        print("\n--- [STL 变换失败] ---")
        return False

# --- 主程序入口 ---
if __name__ == "__main__":
    print("--- [脚本开始 - 最终优化版本 (使用 TFM)] ---")
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="使用 SimpleITK 将 MNI 空间 (LPS) 的 STL 文件通过 TFM 逆变换转换回原始空间 (输出为 RAS 方向)。")
    # 只保留需要的参数
    parser.add_argument('--tfm_inverse_transform', required=True, type=str,
                        help="输入的 ITK TFM 逆变换文件路径 (.tfm)。")
    parser.add_argument('--input_stl', required=True, type=str,
                        help="输入的 STL 文件路径 (MNI 空间, LPS 方向)。")
    parser.add_argument('--output_stl', required=True, type=str,
                        help="输出的 STL 文件路径 (原始空间, RAS 方向)。")

    # 解析参数
    args = parser.parse_args()
    print("\n--- [步骤 0: 解析命令行参数] ---")
    print(f"TFM 逆变换文件: {args.tfm_inverse_transform}")
    print(f"输入 STL 文件:    {args.input_stl}")
    print(f"输出 STL 文件:    {args.output_stl}")

    # 直接调用处理函数
    success = transform_mesh_with_tfm(
        args.input_stl,
        args.output_stl,
        args.tfm_inverse_transform
    )

    if success:
        print("\n--- [脚本执行完毕 - 成功] ---")
    else:
        print("\n--- [脚本执行完毕 - 失败] ---")
        exit(1) # 出错时返回非零退出码
