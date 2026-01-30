import argparse
import trimesh

def convert_stl_to_obj(input_stl_path, output_obj_path):
    """Convert STL file to OBJ format and check mesh validity."""
    # 加载STL文件
    mesh_data = trimesh.load_mesh(input_stl_path)

    # 检查网格是否封闭（水密性）
    if not mesh_data.is_watertight:
        print("Warning: The mesh is not watertight. Exported OBJ may have issues.")

    # 导出为OBJ格式
    mesh_data.export(output_obj_path)
    print(f"Successfully converted {input_stl_path} to {output_obj_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert STL file to OBJ format")
    
    # 定义命令行参数（默认路径需修改为实际路径）
    parser.add_argument("--input_stl", 
                        default="/path/to/your/input.stl", 
                        help="Path to the input .stl file")
    parser.add_argument("--output_obj", 
                        default="/path/to/your/output.obj", 
                        help="Path to the output .obj file")
    
    args = parser.parse_args()
    
    # 执行转换
    convert_stl_to_obj(args.input_stl, args.output_obj)

if __name__ == "__main__":
    main()