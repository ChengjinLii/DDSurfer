import argparse
import trimesh

# Function to convert obj to stl
def convert_obj_to_stl(input_obj_path, output_stl_path):
    # Load the obj file using trimesh
    mesh_data = trimesh.load_mesh(input_obj_path)

    # Check if the mesh is valid
    if not mesh_data.is_watertight:
        print("Warning: The mesh is not watertight. The conversion may not be accurate.")

    # Export the mesh to STL format
    mesh_data.export(output_stl_path)
    print(f"Successfully converted {input_obj_path} to {output_stl_path}")

# Setting up argument parsing
def main():
    parser = argparse.ArgumentParser(description="Convert OBJ file to STL format")
    
    # Define command-line arguments with default paths
    parser.add_argument("--input_obj", default="/path/to/your/file.obj", 
                        help="Path to the input .obj file")
    parser.add_argument("--output_stl", default="/path/to/your/file.stl", 
                        help="Path to the output .stl file")
    
    args = parser.parse_args()
    
    # Call the conversion function
    convert_obj_to_stl(args.input_obj, args.output_stl)

if __name__ == "__main__":
    main()
