### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import open3d as o3d
import numpy as np
import os
import argparse

def largest_cluster(mesh):
    # cluster triangles and find largest surface are cluster
    index_c, count_c, area_c = mesh.cluster_connected_triangles()
    area_c = np.asarray(area_c)
    index_c = np.asarray(index_c)
    area_scale_factor = 0.008 ** 2 / 100.0
    print(f"Found {len(area_c)} clusters, selecting the largest cluster of size {area_c.max()*area_scale_factor:.1f} cmsq. Total initial mesh area is {area_c.sum()*area_scale_factor:.1f} cmsq.")
    # get ind of largest area cluster
    ind_largest_cluster = np.argmax(area_c)
    mask_triangles = index_c == ind_largest_cluster
    mesh.remove_triangles_by_mask(~mask_triangles)

    return mesh

def cut_mesh_at_height(mesh, z_height):
    lower_mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, z_height]))
    upper_mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox([-np.inf, -np.inf, z_height], [np.inf, np.inf, np.inf]))
    
    return upper_mesh, lower_mesh

def cut_and_save_mesh(file_path, output_directory, z_heights):
    # sort z height ascending
    z_heights.sort()
    # binary file
    mesh = o3d.io.read_triangle_mesh(file_path, print_progress=True)
    original_name = os.path.splitext(os.path.basename(file_path))[0]

    previous_mesh = mesh
    cut_counter = 0
    
    for z in z_heights:
        upper_mesh, lower_mesh = cut_mesh_at_height(previous_mesh, z)
        
        upper_file_name = f"cuts/working_thaumato_cut{cut_counter+1}/{original_name}.obj"
        lower_file_name = f"cuts/working_thaumato_cut{cut_counter}/{original_name}.obj"

        os.makedirs(os.path.join(output_directory, "cuts"), exist_ok=True)
        os.makedirs(os.path.join(output_directory, f"cuts/working_thaumato_cut{cut_counter}"), exist_ok=True)
        os.makedirs(os.path.join(output_directory, f"cuts/working_thaumato_cut{cut_counter}/layers"), exist_ok=True)
        lower_mesh = largest_cluster(lower_mesh)
        o3d.io.write_triangle_mesh(os.path.join(output_directory, lower_file_name), lower_mesh)
        if cut_counter >= len(z_heights) - 1:
            os.makedirs(os.path.join(output_directory, f"cuts/working_thaumato_cut{cut_counter+1}"), exist_ok=True)
            os.makedirs(os.path.join(output_directory, f"cuts/working_thaumato_cut{cut_counter+1}/layers"), exist_ok=True)
            upper_mesh = largest_cluster(upper_mesh)
            o3d.io.write_triangle_mesh(os.path.join(output_directory, upper_file_name), upper_mesh)

        previous_mesh = upper_mesh
        cut_counter += 1

def main():
    parser = argparse.ArgumentParser(description="Cut a 3D mesh at specified z-heights and save the parts.")
    parser.add_argument("input_path", type=str, help="Path to the .obj file.")
    parser.add_argument("--output_directory", type=str, default=None, help="Directory where the cut meshes will be saved. Default is the same directory as the input.")
    parser.add_argument("z_heights", type=float, nargs='+', help="List of z-heights to cut the mesh at. For example: 0.2 0.4 0.6")
    
    args = parser.parse_args()

    if args.output_directory is None:
        args.output_directory = os.path.dirname(args.input_path)

    cut_and_save_mesh(args.input_path, args.output_directory, args.z_heights)

if __name__ == "__main__":
    main()
