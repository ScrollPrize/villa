### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2025
"""
Place this script into the surface_pointclouds folder and run it to generate merged point clouds with unique colors for debugging the mask3d pipeline step.
"""

import os
import py7zr
import tarfile
import tempfile
import open3d as o3d
import shutil
import numpy as np

def extract_and_process_files(archive_dir, output_dir):
    """Extracts .7z and .tar files from archive_dir, processes .ply files, and saves merged point clouds to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    
    for archive_file in os.listdir(archive_dir):
        if archive_file.endswith(".7z") or archive_file.endswith(".tar"):
            archive_path = os.path.join(archive_dir, archive_file)
            print(f"Processing: {archive_path}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract archive file
                if archive_file.endswith(".7z"):
                    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
                        archive.extractall(path=temp_dir)
                elif archive_file.endswith(".tar"):
                    with tarfile.open(archive_path, 'r') as archive:
                        archive.extractall(path=temp_dir)
                
                # Load all .ply files and merge with unique colors
                merged_pcd = o3d.geometry.PointCloud()
                color_idx = 0
                colors = np.random.rand(100, 3)  # Generate random colors for up to 100 files
                all_points = []
                
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(".ply"):
                            ply_path = os.path.join(root, file)
                            pcd = o3d.io.read_point_cloud(ply_path)
                            
                            # Assign unique color
                            color_array = np.tile(colors[color_idx % 100], (np.asarray(pcd.points).shape[0], 1))
                            pcd.colors = o3d.utility.Vector3dVector(color_array)
                            
                            merged_pcd += pcd
                            all_points.append(np.asarray(pcd.points))
                            color_idx += 1
                
                # Compute unique points vs total points
                all_points_array = np.vstack(all_points)
                unique_points = np.unique(all_points_array, axis=0)
                total_points = all_points_array.shape[0]
                unique_count = unique_points.shape[0]
                
                print(f"Total points: {total_points}, Unique points: {unique_count}, Overlap: {total_points - unique_count}")
                
                # Save merged point cloud
                output_ply_path = os.path.join(output_dir, os.path.splitext(archive_file)[0] + ".ply")
                o3d.io.write_point_cloud(output_ply_path, merged_pcd)
                print(f"Saved merged point cloud: {output_ply_path}")

if __name__ == "__main__":
    archive_directory = "point_cloud_colorized_verso_subvolume_blocks"  # Change this to your input directory
    output_directory = "debug_instances"  # Change this to your output directory
    extract_and_process_files(archive_directory, output_directory)