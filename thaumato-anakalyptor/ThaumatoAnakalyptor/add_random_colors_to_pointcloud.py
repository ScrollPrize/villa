### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import os
import random
import open3d as o3d
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from threading import current_thread
from multiprocessing import cpu_count

def load_ply(filename):
    """
    Load point cloud data from a .ply file.
    """
    # Load the file and extract the points and normals
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    return points, normals

def save_surface_ply(surface_points, normals, filename):
    # Ensure random colors aren't repeated due to thread-local random state.
    # Seed with the thread's identifier, process id and number of points.
    np.random.seed((len(surface_points) * int((os.getpid() << 16) | (id(current_thread()) & 0xFFFF)))% (2**31))

    # Create an Open3D point cloud object and populate it
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    # random colors
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(surface_points.shape[0], 3)))

    # Save as a PLY file
    o3d.io.write_point_cloud(filename, pcd)

def process_file(file, src_folder, dest_folder):
    # Load volume
    points, normals = load_ply(os.path.join(src_folder, file))
    # Save volume
    save_surface_ply(points, normals, os.path.join(dest_folder, file))


def add_random_colors(src_folder, dest_folder):
    # List all files in the source folder
    all_files = os.listdir(src_folder)

    # Filter out all files that are not .ply files
    ply_files = sorted([file for file in all_files if file.endswith('.ply')])
    
    # Make destination folder if it does not exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Use ThreadPoolExecutor to process files in parallel. ?this does not work on the compute cluster?
    # num_threads = cpu_count()
    # with ProcessPoolExecutor(num_threads) as executor:
    #     list(tqdm(executor.map(process_file, ply_files, [src_folder]*len(ply_files), [dest_folder]*len(ply_files)), total=len(ply_files)))

    # use singlethread
    for file in tqdm(ply_files):
        process_file(file, src_folder, dest_folder)


if __name__ == '__main__':
    # Sample usage:
    src_folder = '/media/julian/SSD2/scroll3_surface_points/point_cloud_recto'  # Replace with your source folder path
    dest_folder = '/media/julian/SSD2/scroll3_surface_points/point_cloud_colorized_recto'  # Replace with your destination folder path

    add_random_colors(src_folder, dest_folder)