import os
import argparse
import numpy as np
import trimesh
from itertools import product


def find_stl_files(root_dir):
    """Recursively find all .stl files in subdirectories."""
    stl_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".stl"):
                stl_files.append(os.path.join(subdir, file))
    return stl_files


def arrange_stls_in_grid(stl_files, spacing=200):
    """Arrange STL files in a square grid and combine them into one mesh."""
    num_files = len(stl_files)
    grid_size = int(np.ceil(np.sqrt(num_files)))

    combined_meshes = []
    positions = product(range(grid_size), repeat=2)

    for i, (x, y) in enumerate(positions):
        if i >= num_files:
            break
        mesh = trimesh.load_mesh(stl_files[i])
        mesh.apply_translation([x * spacing, y * spacing, 0])
        combined_meshes.append(mesh)

    return trimesh.util.concatenate(combined_meshes)


def main():
    parser = argparse.ArgumentParser(
        description="Combine STL files from a directory into a single STL file."
    )
    parser.add_argument(
        "input_dir", type=str, help="Path to the input directory containing STL files."
    )
    args = parser.parse_args()

    output_file = "combined.stl"

    stl_files = find_stl_files(args.input_dir)
    print(f"Found {len(stl_files)} STL files.")

    combined_mesh = arrange_stls_in_grid(stl_files)
    combined_mesh.export(output_file)
    print(f"Exported combined STL as {output_file}")


if __name__ == "__main__":
    main()
