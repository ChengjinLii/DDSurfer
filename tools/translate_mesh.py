import trimesh
import os
import argparse # 导入 argparse 模块
import numpy as np # 导入 numpy 用于向量加法

# --- 参数解析 ---
parser = argparse.ArgumentParser(description="将输入的 OBJ 网格文件按固定的向量平移并保存。")

# 添加输入文件参数
parser.add_argument('--input_obj', required=True, type=str,
                    help="需要平移的输入 OBJ 文件的完整路径。")

# 添加输出文件参数
parser.add_argument('--output_obj', required=True, type=str,
                    help="平移后要保存的 OBJ 文件的完整路径。")

# 解析命令行参数
args = parser.parse_args()

# 从参数中获取文件路径
input_file = args.input_obj
output_file = args.output_obj
# --- 参数解析结束 ---

# --- 核心处理逻辑 ---
# 定义固定的平移向量
translation_vector = np.array([85, 132, 70]) # 使用 numpy 数组以便于向量加法

print(f"准备处理输入文件: {input_file}")
print(f"平移向量: {translation_vector}")
print(f"输出文件将保存至: {output_file}")

# 检查输入文件是否存在
if not os.path.exists(input_file):
    print(f"错误：找不到输入文件 {input_file}。脚本将退出。")
    exit(1) # 如果输入文件不存在，则退出

# 加载、平移并保存网格
try:
    # 加载网格，process=False 避免 trimesh 自动处理（如合并顶点）
    print("正在加载网格...")
    mesh = trimesh.load(input_file, process=False)
    print(f"网格加载成功，包含 {len(mesh.vertices)} 个顶点和 {len(mesh.faces)} 个面。")

    # 执行平移操作：将平移向量加到每个顶点坐标上
    print("正在平移顶点...")
    translated_vertices = mesh.vertices + translation_vector

    # 使用平移后的顶点和原始面信息创建新的 Trimesh 对象
    translated_mesh = trimesh.Trimesh(vertices=translated_vertices, faces=mesh.faces, process=False)
    print("平移完成，已创建新网格对象。")

    # 获取输出文件的目录路径
    output_dir = os.path.dirname(output_file)

    # 如果输出目录非空且不存在，则创建它
    # os.path.dirname 对于仅有文件名的情况会返回空字符串
    if output_dir and not os.path.exists(output_dir):
        print(f"输出目录 {output_dir} 不存在，正在创建...")
        os.makedirs(output_dir)

    # 导出（保存）平移后的网格到指定的输出文件
    print(f"正在保存平移后的网格到 {output_file}...")
    # 使用 file_type='obj' 确保保存为 OBJ 格式
    translated_mesh.export(output_file, file_type='obj')
    print(f"平移后的网格已成功保存。")

except Exception as e:
    print(f"处理文件 {input_file} 时发生错误: {e}")
    exit(1) # 如果处理过程中出错，则退出

print("\n脚本执行完毕。")