### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import numpy as np
from pathlib import Path
import os
import open3d as o3d
import multiprocessing
from functools import partial
import argparse

def save_point_cloud(points, normals, labels, filename="point_cloud.txt"):
    """
    Save point cloud in .txt format for use with the process_file function.
    """

    # Convert labels into processed labels and instance ids.
    processed_labels = np.where(labels < 0, 0, 1)
    instance_ids = np.where(labels < 0, -100, labels)  # Changed -1 to -100 directly
    colors = np.random.rand(*points.shape)

    # Assuming the points array shape is (n, 3) and normals is (n, 3),
    # concatenate them along with labels to form the desired structure.
    # Here, we'll only use the first three columns of normals for now.
    data_to_save = np.hstack((points, colors, processed_labels[:, None], instance_ids[:, None]))

    # Convert filename to Path object for compatibility and save as .txt format.
    filepath = Path(filename)

    # Create parent directories if they don't exist.
    filepath.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(filepath, data_to_save, delimiter=',')  # Using comma as delimiter

def load_ply(filename):
    """
    Load point cloud data from a .ply file.
    """
    # Check that the file exists
    assert os.path.isfile(filename), f"File {filename} not found."

    # Load the file and extract the points and normals
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    return points, normals

def generate_training_sample_from_annotation(src_folder, dest_folder, volume_folder=None):
    print(f"Processing {src_folder}")
    # Get volume in src_folder with name *.ply
    ply_files = [file for file in os.listdir(src_folder) if file.endswith('.ply')]
    ply_file = ply_files[0]
    ply_file_path = os.path.join(src_folder, ply_file)
    # Load volume
    points, normals = load_ply(ply_file_path)
    # Try to reload volume from volume_folder if provided
    if volume_folder is not None:
        # find ply_file in volume_folder
        ply_file_path = os.path.join(volume_folder, ply_file)
        try:
            # Load volume
            points, normals = load_ply(ply_file_path)
        except:
            print(f"Could not load {ply_file_path}")
            pass

    labels = np.zeros(points.shape[0]) - 1  # All points are background

    # All .ply files in "surfaces" folder are annotations. Load them.
    surfaces_folder = os.path.join(src_folder, "surfaces")
    surface_files = [file for file in os.listdir(surfaces_folder) if file.endswith('.ply')]

    for i, surface_file in enumerate(surface_files):
        # Load surface
        surface_file_path = os.path.join(surfaces_folder, surface_file)
        surface_points, surface_normals = load_ply(surface_file_path)

        # Find the closest point in the volume for each point in the surface
        distances = np.linalg.norm(points[:, None, :] - surface_points[None, :, :], axis=-1)
        closest_points_indices = np.argmin(distances, axis=0)
        closest_points_distance = np.min(distances, axis=0)

        # Check that the closest point is within a threshold distance
        threshold = 0.1
        assert np.all(closest_points_distance < threshold)       

        # Label the closest points with the surface label
        labels[np.argmin(distances, axis=0)] = i

    # Normalize points to be in the range [0.0, 50.0] in all axes
    points = points - np.min(points, axis=0)
    points = points / np.max(points, axis=0)
    points = points * 50.0

    # Save the point cloud 
    txt_file = ply_file.replace(".ply", ".txt")
    save_point_cloud(points, normals, labels, os.path.join(dest_folder, txt_file))

    return points, normals, labels, txt_file

def process_folder(sub_folder, src_folder, dest_folder, volume_folder):    
    # dest folder for each sub_folder
    dest_sub_folder = os.path.join(dest_folder, sub_folder)
    res = generate_training_sample_from_annotation(os.path.join(src_folder, sub_folder), dest_sub_folder, volume_folder)
    if res is None:
        print(f"Sample {sub_folder} failed")
        return
    points, normals, labels, txt_file = res
    # save all .txt also in a data folder
    data_folder = os.path.join(dest_folder, "data")
    save_point_cloud(points, normals, labels, os.path.join(data_folder, txt_file))


def generate_training_samples_from_annotations(src_folder, dest_folder, volume_folder=None):
    # get all folders in src_folder
    all_folders = os.listdir(src_folder)

    # Create a multiprocessing Pool and execute function on multiple processes
    with multiprocessing.Pool(processes=1) as pool:
        pool.map(partial(process_folder, src_folder=src_folder, dest_folder=dest_folder, volume_folder=volume_folder), all_folders)

if __name__ == "__main__":
    src_folder = 'ren 60 images'  # Replace with your source folder path
    volume_folder = None # 'point_cloud_Ren_run'  # Replace with your original volume folder path # Volume to extract original PointCloud from. Run integrity checks on human annotations
    dest_folder = 'annotations_Ren'  # Replace with your destination folder path

    parser = argparse.ArgumentParser(description='Generate training samples from annotations')
    parser.add_argument('--src_folder', type=str, default=src_folder, help='Source folder path')
    parser.add_argument('--dest_folder', type=str, default=dest_folder, help='Destination folder path')
    parser.add_argument('--volume_folder', type=str, default=volume_folder, help='Volume folder path, runs integrity checks on human annotations')

    generate_training_samples_from_annotations(src_folder, dest_folder, volume_folder)